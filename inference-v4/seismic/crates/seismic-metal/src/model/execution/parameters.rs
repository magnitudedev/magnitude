//! Integer facts over original compile-time parameters. Runtime controls remain
//! invocation inputs and cannot be turned into choices by this interpretation.
use super::*;
use magnitude_solver::model::{Constraint, Domain, Literal, ModelBuilder, VarId};
use seismic_accounting::algebra::Value;
use seismic_compiler::tuner::expressions::Expressions;
use seismic_lang::sym::Sym;
use std::collections::HashMap;
mod inverse;

#[derive(Clone, PartialEq, Eq)]
pub(super) enum Predicate {
    Constant(bool),
    Nonnegative(Sym),
    And(Vec<Predicate>),
    Not(Box<Predicate>),
}
#[derive(Clone, PartialEq)]
pub(super) struct Definition {
    integers: [Option<Sym>; 32],
    predicates: [Option<Predicate>; 32],
}
#[derive(Clone, PartialEq)]
struct DeferredDefinition {
    definition: Definition,
    values: Option<Values>,
    ranges: Option<ranges::Ranges>,
    affine: Option<affine::Values>,
}
pub(super) struct Participation {
    pub active: VarId,
    pub mask: u32,
    coordinate: Option<(Sym, i64, i64)>,
}
#[derive(Clone, Default)]
pub(super) struct Symbols {
    pub parameters: BTreeMap<String, Value>,
    /// Unconditional definitions expanded to the original independent operands.
    /// The Value registry still owns every original solver variable and bound.
    definitions: BTreeMap<String, Sym>,
    pub locals: BTreeMap<String, [Option<Sym>; 32]>,
    predicates: BTreeMap<String, [Option<Predicate>; 32]>,
    intervals: Vec<(Sym, i64, i64)>,
    enclosure: inverse::Enclosure,
    // Expanded symbols depend only on this immutable participation context.
    // Branch snapshots share its answers; changing assumptions starts a fresh cache.
    bounds_cache: std::rc::Rc<std::cell::RefCell<HashMap<Sym, Option<(i64, i64)>>>>,
    refinement_pending: bool,
    conditional: BTreeMap<String, Vec<(Vec<Literal>, DeferredDefinition)>>,
    choice_cardinalities: BTreeMap<VarId, usize>,
}
impl Symbols {
    pub fn with_definitions(parameters: BTreeMap<String, Value>, definitions: BTreeMap<String, Sym>) -> Result<Self, DerivationError> {
        fn expand(name: &str, parameters: &BTreeMap<String, Value>, definitions: &BTreeMap<String, Sym>,
            visiting: &mut BTreeSet<String>, expanded: &mut BTreeMap<String, Sym>) -> Result<Sym, DerivationError> {
            if !parameters.contains_key(name) { return Err(format!("Metal numeric definition references unbound parameter {name}").into()); }
            if let Some(value) = expanded.get(name) { return Ok(value.clone()); }
            let Some(definition) = definitions.get(name) else { return Ok(Sym::param(name)); };
            if !visiting.insert(name.into()) { return Err(format!("cyclic Metal numeric definition for {name}").into()); }
            let mut replacements = HashMap::new();
            for dependency in definition.params() {
                let replacement = expand(&dependency, parameters, definitions, visiting, expanded)?;
                replacements.insert(dependency, replacement);
            }
            let value = seismic_lang::lower::subst_sym(definition, &replacements, &HashMap::new());
            visiting.remove(name);
            expanded.insert(name.into(), value.clone());
            Ok(value)
        }
        let mut expanded = BTreeMap::new();
        let mut visiting = BTreeSet::new();
        for name in definitions.keys() { expand(name, &parameters, &definitions, &mut visiting, &mut expanded)?; }
        // The retained bounds on a derived operand also constrain its defining
        // expression. Keep them as immutable premises in every guarded context.
        let intervals = expanded.iter().filter_map(|(name, expression)| {
            let (low, high) = parameters.get(name)?.bounds();
            Some((expression.clone(), i64::try_from(low).ok()?, i64::try_from(high).ok()?))
        }).collect::<Vec<_>>();
        let enclosure = inverse::Enclosure::derive(&parameters, &intervals);
        Ok(Self { parameters, definitions: expanded, intervals, enclosure, ..Self::default() })
    }
    fn parameter(&self, name: &str) -> Sym {
        self.definitions.get(name).cloned().unwrap_or_else(|| Sym::param(name))
    }
    fn parameter_affine(&self, name: &str, symbol: &Sym, range: (i64, i64)) -> Option<affine::Value> {
        self.affine(symbol).or_else(|| {
            // A nonlinear definition still names this exact original operand.
            // Retain its identity for common address translations; importing the
            // coordinate back into Symbols restores the defining expression.
            let binding = self.parameters.get(name)?;
            Some(affine::Value::coordinate((1u64 << 63) | binding.id().0 as u64,
                i128::from(range.0), i128::from(range.1)))
        })
    }
    /// Make current guarded facts available to the ordinary typed interpreter,
    /// including helper arguments that are expressions rather than bindings.
    pub fn annotate_statement(&self, state: &mut Derivation<'_>, statement: &Statement) {
        use Statement as S;
        match statement {
            S::Let { value, .. } | S::Assign { value, .. } | S::Evaluate(value)
            | S::If(value) | S::ReturnIf(value) | S::Return(Some(value)) => self.annotate(state, value),
            S::Pointer { index, .. } | S::VectorRead { index, .. } => self.annotate(state, index),
            S::Write { index, value, .. } => { self.annotate(state, index); self.annotate(state, value); },
            S::For { start, end, .. } => { self.annotate(state, start); self.annotate(state, end); },
            S::MatrixLoad { offset, leading, .. } | S::MatrixStore { offset, leading, .. } => {
                self.annotate(state, offset); self.annotate(state, leading);
            },
            S::Array { .. } | S::Else | S::Scope | S::End | S::Return(None)
            | S::FailureStatus | S::Barrier | S::Fragment { .. }
            | S::MatrixMultiplyAccumulate { .. } | S::Unmapped(_) => {},
        }
    }
    pub fn annotate(&self, state: &mut Derivation<'_>, expression: &Expression) {
        use Expression as E;
        match expression {
            E::Binary(_, left, right, _) | E::ShortCircuit { left, right, .. } => {
                self.annotate(state, left); self.annotate(state, right);
            },
            E::Unary(_, value, _) | E::Cast(_, value) | E::Bitcast(_, value)
            | E::Read { index: value, .. } => self.annotate(state, value),
            E::Builtin(_, arguments, _) | E::Helper(_, arguments, _) => {
                for argument in arguments { self.annotate(state, argument); }
            },
            E::Select(condition, yes, no) | E::EagerSelect(condition, yes, no) => {
                self.annotate(state, condition); self.annotate(state, yes); self.annotate(state, no);
            },
            E::Integer(..) | E::Float(..) | E::Variable(..) | E::VectorElement { .. }
            | E::Parameter { .. } | E::Unmapped(..) => {},
        }
        let mut facts = state.expression_facts(expression);
        for lane in 0..32 {
            if state.active & (1 << lane) == 0 { continue; }
            if let Some(symbol) = self.expression(state, expression, lane) {
                if let Some((lo, hi)) = self.bounds(&symbol) {
                    facts.ranges[lane] = Some((i128::from(lo), i128::from(hi)));
                    facts.affine[lane] = if lo == hi { Some(affine::Value::constant(i128::from(lo))) }
                        else { self.affine(&symbol).or_else(|| facts.affine[lane].clone()) };
                }
            } else if expression.ty() == Type::Bool {
                if let Some(value) = self.formula(state, expression, lane).and_then(|value| self.truth(&value)) {
                    let value = i128::from(value);
                    facts.ranges[lane] = Some((value, value));
                    facts.affine[lane] = Some(affine::Value::constant(value));
                }
            }
        }
        state.assumptions.insert(expression.clone(), facts);
    }
    pub fn assume(&mut self, state: &mut Derivation<'_>, guards: &[Literal]) {
        state.assumptions.clear();
        let selected = self.conditional.iter().filter_map(|(name, alternatives)| {
            crate::model::memory::assumed_alternative(alternatives, guards, &self.choice_cardinalities)
                .map(|definition| (name.clone(), definition.clone()))
        }).collect::<Vec<_>>();
        self.refinement_pending |= !selected.is_empty();
        for (name, definition) in selected {
            self.conditional.remove(&name);
            self.locals.insert(name.clone(), definition.definition.integers);
            self.predicates.insert(name.clone(), definition.definition.predicates);
            match definition.values { Some(value) => { state.env.insert(name.clone(), value); }, None => { state.env.remove(&name); } }
            match definition.ranges { Some(value) => { state.ranges.insert(name.clone(), value); }, None => { state.ranges.remove(&name); } }
            match definition.affine { Some(value) => { state.affine.insert(name.clone(), value); }, None => { state.affine.remove(&name); } }
        }
        if self.refinement_pending {
            self.refine_facts(state);
            self.refinement_pending = false;
        }
        state.facts.clear();
    }
    fn refine_facts(&self, state: &mut Derivation<'_>) {
        for name in self.parameters.keys() {
            let symbol = self.parameter(name);
            let Some((lo, hi)) = self.bounds(&symbol) else { continue; };
            state.ranges.insert(name.clone(), [Some((i128::from(lo), i128::from(hi))); 32]);
            state.env.insert(name.clone(), [(lo == hi).then_some(lo as u64); 32]);
            let affine = self.parameter_affine(name, &symbol, (lo, hi));
            state.affine.insert(name.clone(), std::array::from_fn(|_| affine.clone()));
        }
        for name in self.locals.keys() {
            for lane in 0..32 {
                let symbol = self.locals.get(name).and_then(|values| values[lane].clone());
                let Some(symbol) = symbol else { continue; };
                let Some((lo, hi)) = self.bounds(&symbol) else { continue; };
                state.ranges.entry(name.clone()).or_insert([None; 32])[lane] = Some((i128::from(lo), i128::from(hi)));
                state.env.entry(name.clone()).or_insert([None; 32])[lane] = (lo == hi).then_some(lo as u64);
                state.affine.entry(name.clone()).or_insert_with(|| std::array::from_fn(|_| None))[lane] =
                    self.affine(&symbol);
            }
        }
    }
    /// Keep branch-local integer and address facts attached to the original
    /// implementation guard, just like retained native pointer definitions.
    pub fn join_alternative(&mut self, other: &Self, env: &BTreeMap<String, Values>,
        ranges: &BTreeMap<String, ranges::Ranges>, affine: &BTreeMap<String, affine::Values>,
        joined_env: &BTreeMap<String, Values>, joined_ranges: &BTreeMap<String, ranges::Ranges>,
        joined_affine: &BTreeMap<String, affine::Values>, guards: &[Literal]) {
        let names = other.locals.keys().chain(other.predicates.keys()).collect::<BTreeSet<_>>();
        for name in names {
            if self.locals.get(name) == other.locals.get(name) && self.predicates.get(name) == other.predicates.get(name)
                && joined_env.get(name) == env.get(name) && joined_ranges.get(name) == ranges.get(name)
                && joined_affine.get(name) == affine.get(name) { continue; }
            let definition = DeferredDefinition {
                definition: Definition {
                    integers: other.locals.get(name).cloned().unwrap_or_else(|| std::array::from_fn(|_| None)),
                    predicates: other.predicates.get(name).cloned().unwrap_or_else(|| std::array::from_fn(|_| None)),
                },
                values: env.get(name).copied(), ranges: ranges.get(name).copied(), affine: affine.get(name).cloned(),
            };
            let output = self.conditional.entry(name.clone()).or_default();
            crate::model::memory::retain_alternative(output, guards.to_vec(), definition);
        }
        for (name, alternatives) in &other.conditional {
            let output = self.conditional.entry(name.clone()).or_default();
            for (activation, definition) in alternatives {
                let mut activation = activation.clone(); activation.extend_from_slice(guards);
                crate::model::memory::retain_alternative(output, activation, definition.clone());
            }
        }
    }
    pub fn collapse_choice(&mut self, variable: VarId, cardinality: usize) {
        self.choice_cardinalities.insert(variable, cardinality);
        for alternatives in self.conditional.values_mut() {
            crate::model::memory::collapse_alternatives(alternatives, variable, cardinality);
        }
    }
    pub fn initialize(&self, state: &mut Derivation<'_>) {
        for (name, value) in &self.parameters {
            let symbol = self.parameter(name);
            let range = self.bounds(&symbol);
            let (low, high) = range.map(|(low, high)| (i128::from(low), i128::from(high)))
                .unwrap_or((i128::from(value.bounds().0), i128::from(value.bounds().1)));
            state.env.insert(name.clone(), [(low == high).then_some(low as u64); 32]);
            state.ranges.insert(name.clone(), [Some((low, high)); 32]);
            let affine = range.and_then(|range| self.parameter_affine(name, &symbol, range));
            state.affine.insert(name.clone(), std::array::from_fn(|_| affine.clone()));
        }
    }
    fn expression(&self, state: &Derivation<'_>, expression: &Expression, lane: usize) -> Option<Sym> {
        use Expression as E;
        let recurse = |expression: &E| self.expression(state, expression, lane);
        let symbol = match expression {
            E::Integer(value, _) => Sym::constant(*value),
            E::Variable(name, ty) | E::Parameter { name, ty } => {
                if self.parameters.contains_key(name) { self.parameter(name) }
                else if let Some(value) = self.locals.get(name).and_then(|values| values[lane].clone()) { value }
                else {
                    let value = state.env.get(name)?[lane]?;
                    let value = match ty { Type::I32 => i64::from(value as i32), Type::U32 => i64::from(value as u32), Type::I64 => value as i64,
                        Type::U64 => i64::try_from(value).ok()?, Type::Bool => i64::from(value != 0), _ => return None };
                    Sym::constant(value)
                }
            },
            E::Cast(_, value) if value.ty() == Type::Bool => {
                self.indicator(&self.formula(state, value, lane)?)?
            },
            E::Cast(_, value) => recurse(value)?,
            // Signed wrapping arithmetic is emitted through equal-width
            // unsigned bitcasts. It preserves the integer symbol only when
            // the final range check below proves the value fits both views.
            E::Bitcast(ty, value) if ty.bytes() == value.ty().bytes() => recurse(value)?,
            E::Select(condition, yes, no) | E::EagerSelect(condition, yes, no) => {
                let selected = self.truth(&self.formula(state, condition, lane)?)?;
                recurse(if selected { yes } else { no })?
            },
            E::Helper(helper, arguments, element) => {
                recurse(&crate::terminal::helper::single_expression(*helper, arguments, *element)?)?
            },
            E::Builtin(name, arguments, _) if matches!(name.as_str(), "min" | "max") && arguments.len() == 2 => {
                let left = recurse(&arguments[0])?; let right = recurse(&arguments[1])?;
                let (minimum, maximum) = self.bounds(&left.sub(&right))?;
                if maximum <= 0 { if name == "min" { left } else { right } }
                else if minimum >= 0 { if name == "min" { right } else { left } }
                else { return None; }
            },
            E::Unary(UnaryOp::Neg, value, _) => Sym::constant(0).sub(&recurse(value)?),
            E::Binary(operation, left, right, _) => {
                let left = recurse(left)?; let right = recurse(right)?;
                match operation {
                    BinaryOp::Add => left.add(&right), BinaryOp::Sub => left.sub(&right), BinaryOp::Mul => left.mul(&right),
                    BinaryOp::Div | BinaryOp::Rem => {
                        let numerator = self.bounds(&left)?; let divisor = self.bounds(&right)?;
                        // Target integer division truncates. Nonnegative inputs
                        // agree with the source family's Euclidean equations.
                        if numerator.0 < 0 || divisor.0 <= 0 { return None; }
                        let quotient = left.quot(&right);
                        let (first, last) = self.bounds(&quotient)?;
                        if first == last {
                            if *operation == BinaryOp::Div { Sym::constant(first) }
                            else { left.sub(&right.mul(&Sym::constant(first))) }
                        } else if *operation == BinaryOp::Div { quotient }
                        else { left.rem(&right) }
                    },
                    _ => return None,
                }
            },
            _ => return None,
        };
        let (lo, hi) = self.bounds(&symbol)?;
        ranges::fit((i128::from(lo), i128::from(hi)), expression.ty())?;
        Some(symbol)
    }
    fn bounds(&self, expression: &Sym) -> Option<(i64, i64)> {
        if let Some(bounds) = self.bounds_cache.borrow().get(expression) { return *bounds; }
        let bounds = expression.eval_interval_with(&|name| {
            self.enclosure.parameters.get(name).copied()
        }, &|expression| {
            let mut bound: Option<(i64, i64)> = None;
            for (coordinate, low, high) in &self.intervals {
                if let Some(offset) = expression.sub(coordinate).as_constant() {
                    let low = i64::try_from(i128::from(*low) + i128::from(offset)).ok()?;
                    let high = i64::try_from(i128::from(*high) + i128::from(offset)).ok()?;
                    bound = Some(bound.map_or((low, high), |(lo, hi)| (lo.max(low), hi.min(high))));
                } else if let Some(sum) = expression.add(coordinate).as_constant() {
                    let lower = i64::try_from(i128::from(sum) - i128::from(*high)).ok()?;
                    let upper = i64::try_from(i128::from(sum) - i128::from(*low)).ok()?;
                    bound = Some(bound.map_or((lower, upper), |(lo, hi)| (lo.max(lower), hi.min(upper))));
                }
            }
            bound
        });
        self.bounds_cache.borrow_mut().insert(expression.clone(), bounds);
        bounds
    }
    /// Only a proved contradiction makes a guarded analysis region absent.
    /// Unknown arithmetic remains an analysis gap and cannot reject a choice.
    pub fn inconsistent(&self) -> bool {
        if self.enclosure.inconsistent { return true; }
        self.intervals.iter().any(|(coordinate, low, high)| {
            coordinate.eval_interval(&|name| self.enclosure.parameters.get(name).copied())
                .is_some_and(|(lo, hi)| hi < *low || lo > *high)
        })
    }
    /// Return the complete symbolic participation formula for every active
    /// lane. Keeping this per lane is useful to address analysis: a lane may
    /// be active for a parameter guard even when another lane is inactive, and
    /// collapsing the mask to its first lane would silently drop transactions.
    pub fn participation(&self, state: &Derivation<'_>, expression: &Expression) -> [Option<Predicate>; 32] {
        std::array::from_fn(|lane| {
            if state.active & (1 << lane) == 0 { None } else { self.formula(state, expression, lane) }
        })
    }
    /// A family of lane masks induced by one symbolic threshold coordinate.
    /// Coordinate values are never enumerated: at most one interval per lane
    /// boundary is needed, regardless of the numeric parameter's domain size.
    /// Boolean combinations preserve exact masks on each interval.
    pub fn participation_regions(&self, state: &Derivation<'_>, builder: &mut ModelBuilder,
        expression: &Expression) -> Result<Option<Vec<Participation>>, DerivationError> {
        let predicates = self.participation(state, expression);
        self.participation_regions_for(state.active, builder, &predicates)
    }
    pub fn participation_regions_for(&self, active_lanes: u32, builder: &mut ModelBuilder,
        predicates: &[Option<Predicate>; 32]) -> Result<Option<Vec<Participation>>, DerivationError> {
        if (0..32).any(|lane| active_lanes & (1 << lane) != 0 && predicates[lane].is_none()) { return Ok(None); }
        fn collect(symbols: &Symbols, predicate: &Predicate, atoms: &mut Vec<Sym>) {
            if symbols.truth(predicate).is_some() { return; }
            match predicate {
                Predicate::Nonnegative(value) => atoms.push(value.clone()),
                Predicate::And(values) => for value in values { collect(symbols, value, atoms); },
                Predicate::Not(value) => collect(symbols, value, atoms),
                Predicate::Constant(_) => {},
            }
        }
        fn evaluate(symbols: &Symbols, predicate: &Predicate, coordinate: Option<(&Sym, i64)>) -> Option<bool> {
            if let Some(value) = symbols.truth(predicate) { return Some(value); }
            Some(match predicate {
                Predicate::Constant(value) => *value,
                Predicate::Nonnegative(value) => {
                    let (base, at) = coordinate?;
                    let offset = value.sub(base).as_constant()?;
                    i128::from(at) + i128::from(offset) >= 0
                },
                Predicate::Not(value) => !evaluate(symbols, value, coordinate)?,
                Predicate::And(values) => {
                    let mut result = true;
                    for value in values { result &= evaluate(symbols, value, coordinate)?; }
                    result
                },
            })
        }
        let mut atoms = Vec::new();
        for (lane, predicate) in predicates.iter().enumerate() {
            if active_lanes & (1 << lane) != 0 {
                if let Some(predicate) = predicate { collect(self, predicate, &mut atoms); }
            }
        }
        let mut next = builder.clone();
        let mut regions = Vec::new();
        if let Some(coordinate) = atoms.first() {
            let Some((minimum, maximum)) = self.bounds(coordinate) else { return Ok(None); };
            let mut boundaries = std::collections::BTreeSet::from([minimum]);
            for atom in &atoms {
                let Some(offset) = atom.sub(coordinate).as_constant() else { return Ok(None); };
                let boundary = -i128::from(offset);
                if boundary > i128::from(minimum) && boundary <= i128::from(maximum) { boundaries.insert(boundary as i64); }
            }
            let boundaries = boundaries.into_iter().collect::<Vec<_>>();
            let mut expressions = Expressions::new(self.parameters.clone());
            for (index, &begin) in boundaries.iter().enumerate() {
                let end = boundaries.get(index + 1).map_or(maximum, |next| next - 1);
                let mut mask = 0u32;
                for lane in 0..32 {
                    if active_lanes & (1 << lane) == 0 { continue; }
                    let Some(live) = evaluate(self, predicates[lane].as_ref().unwrap(), Some((coordinate, begin))) else { return Ok(None); };
                    if live { mask |= 1 << lane; }
                }
                let lower = expressions.predicate(&mut next, "metal.participation.lower", &coordinate.sub(&Sym::constant(begin)))
                    .map_err(|error| DerivationError::Unsupported(error.to_string()))?;
                let upper = expressions.predicate(&mut next, "metal.participation.upper", &Sym::constant(end).sub(coordinate))
                    .map_err(|error| DerivationError::Unsupported(error.to_string()))?;
                let active = next.local_variable("metal.participation.interval", Domain::boolean()).map_err(|error| error.to_string())?;
                next.constraint(Constraint::BoolAnd { output: active, inputs: vec![lower, upper] });
                regions.push(Participation { active, mask, coordinate: Some((coordinate.clone(), begin, end)) });
            }
        } else {
            let mut mask = 0u32;
            for lane in 0..32 {
                if active_lanes & (1 << lane) == 0 { continue; }
                let Some(live) = evaluate(self, predicates[lane].as_ref().unwrap(), None) else { return Ok(None); };
                if live { mask |= 1 << lane; }
            }
            let active = next.local_variable("metal.participation.constant", Domain::singleton(1)).map_err(|error| error.to_string())?;
            regions.push(Participation { active, mask, coordinate: None });
        }
        *builder = next;
        Ok(Some(regions))
    }
    pub fn assume_participation(&mut self, region: &Participation) {
        if let Some(coordinate) = &region.coordinate {
            self.intervals.push(coordinate.clone());
            self.enclosure = inverse::Enclosure::derive(&self.parameters, &self.intervals);
            self.bounds_cache = Default::default();
            self.refinement_pending = true;
        }
    }
    pub fn interval(&self, state: &Derivation<'_>, expression: &Expression) -> Option<(i64, i64)> {
        (0..32).filter(|lane| state.active & (1 << lane) != 0).try_fold(None, |bounds, lane| {
            let (lo, hi) = self.bounds(&self.expression(state, expression, lane)?)?;
            Some(Some(match bounds { None => (lo, hi), Some((a, b)) => (lo.min(a), hi.max(b)) }))
        }).flatten()
    }
    pub fn offset(&self, definition: &Definition, amount: i64) -> Definition {
        Definition { integers: std::array::from_fn(|lane| definition.integers[lane].as_ref().map(|value| value.add(&Sym::constant(amount)))),
            predicates: std::array::from_fn(|_| None) }
    }
    pub fn values(&self, state: &Derivation<'_>, expression: &Expression) -> Definition {
        Definition { integers: std::array::from_fn(|lane| self.expression(state, expression, lane)),
            predicates: std::array::from_fn(|lane| self.formula(state, expression, lane)) }
    }
    pub fn assign(&mut self, state: &mut Derivation<'_>, name: &str, definition: Definition) {
        let Definition {integers: mut symbols, predicates: mut predicates} = definition;
        for lane in 0..32 {
            if state.active & (1 << lane) == 0 {
                symbols[lane] = self.locals.get(name).and_then(|values| values[lane].clone());
                predicates[lane] = self.predicates.get(name).and_then(|values| values[lane].clone());
            } else if symbols[lane].is_none() {
                // Typed helper interpretation already established the returned
                // value's provenance. Import that fact instead of duplicating
                // helper semantics in the symbolic interpreter.
                symbols[lane] = state.affine.get(name).and_then(|values| values[lane].as_ref())
                    .and_then(|value| self.from_affine(value));
            }
        }
        if let Some(values) = state.env.get_mut(name) {
            for lane in 0..32 {
                if values[lane].is_none() && state.active & (1 << lane) != 0 {
                    values[lane] = symbols[lane].as_ref().and_then(|value| self.bounds(value))
                        .filter(|(lo, hi)| lo == hi).map(|(value, _)| value as u64)
                        .or_else(|| predicates[lane].as_ref().and_then(|value| self.truth(value)).map(u64::from));
                }
            }
        }
        self.predicates.insert(name.into(), predicates);
        // The ordinary typed interpreter may establish facts that are not
        // expressible in the compile-time polynomial vocabulary (for example a
        // checked helper or a bounded indirect read). Do not erase those facts.
        let facts = std::array::from_fn(|lane| symbols[lane].as_ref().and_then(|symbol| self.bounds(symbol))
            .map(|(lo, hi)| (i128::from(lo), i128::from(hi)))
            .or_else(|| state.ranges.get(name).and_then(|values| values[lane])));
        let affine = std::array::from_fn(|lane| symbols[lane].as_ref().and_then(|symbol| self.affine(symbol))
            .or_else(|| state.affine.get(name).and_then(|values| values[lane].clone())));
        self.locals.insert(name.into(), symbols);
        state.ranges.insert(name.into(), facts); state.affine.insert(name.into(), affine); state.facts.clear();
    }
    fn from_affine(&self, value: &affine::Value) -> Option<Sym> {
        let mut symbol = Sym::constant(i64::try_from(value.base).ok()?);
        for (&coordinate, &(coefficient, _)) in &value.terms {
            let (name, _) = self.parameters.iter().find(|(_, parameter)|
                coordinate == ((1u64 << 63) | parameter.id().0 as u64))?;
            symbol = symbol.add(&self.parameter(name).mul(&Sym::constant(i64::try_from(coefficient).ok()?)));
        }
        Some(symbol)
    }
    fn affine(&self, symbol: &Sym) -> Option<affine::Value> {
        let mut output = affine::Value::constant(0);
        for (monomial, coefficient) in symbol.monomials() {
            if monomial.is_empty() { output = output.add(affine::Value::constant(i128::from(coefficient)))?; continue; }
            if monomial.len() != 1 { return None; }
            let (seismic_lang::sym::Atom::Param(name), &1) = monomial.iter().next()? else { return None; };
            let binding = self.parameters.get(name)?;
            let coordinate = (1u64 << 63) | binding.id().0 as u64;
            let (minimum, maximum) = self.bounds(&Sym::param(name))?;
            let value = affine::Value::coordinate(coordinate, i128::from(minimum), i128::from(maximum))
                .scale(i128::from(coefficient))?;
            output = output.add(value)?;
        }
        Some(output)
    }
    pub fn remove(&mut self, name: &str) { self.locals.remove(name); self.predicates.remove(name); self.conditional.remove(name); }
    pub fn restore_local(&mut self, name: &str, before: &Self) {
        match before.locals.get(name) { Some(value) => { self.locals.insert(name.into(), value.clone()); }, None => { self.locals.remove(name); } }
        match before.predicates.get(name) { Some(value) => { self.predicates.insert(name.into(), value.clone()); }, None => { self.predicates.remove(name); } }
        match before.conditional.get(name) { Some(value) => { self.conditional.insert(name.into(), value.clone()); }, None => { self.conditional.remove(name); } }
    }
    pub fn join(&mut self, other: &Self) {
        self.choice_cardinalities.extend(other.choice_cardinalities.iter().map(|(&variable, &cardinality)| (variable, cardinality)));
        self.conditional.retain(|name, alternatives| {
            alternatives.retain(|alternative| other.conditional.get(name).is_some_and(|right| right.contains(alternative)));
            !alternatives.is_empty()
        });
        let previous_intervals = self.intervals.len();
        self.intervals.retain(|interval| other.intervals.contains(interval));
        self.enclosure = inverse::Enclosure::derive(&self.parameters, &self.intervals);
        if previous_intervals != self.intervals.len() {
            self.bounds_cache = Default::default();
            self.refinement_pending = true;
        }
        self.predicates.retain(|name, left| {
            let Some(right) = other.predicates.get(name) else { return false; };
            for lane in 0..32 { if left[lane] != right[lane] { left[lane] = None; } } true
        });
        self.locals.retain(|name, left| {
            let Some(right) = other.locals.get(name) else { return false; };
            for lane in 0..32 { if left[lane] != right[lane] { left[lane] = None; } } true
        });
    }
    fn truth(&self, predicate: &Predicate) -> Option<bool> {
        match predicate {
            Predicate::Constant(value) => Some(*value),
            Predicate::Nonnegative(value) => { let (lo, hi) = self.bounds(value)?; if lo >= 0 {Some(true)} else if hi < 0 {Some(false)} else {None} },
            Predicate::Not(value) => self.truth(value).map(|value| !value),
            Predicate::And(values) => {
                let mut known = true;
                for value in values { match self.truth(value) {Some(false) => return Some(false), Some(true) => {}, None => known = false} }
                known.then_some(true)
            },
        }
    }
    /// Exact 0/1 arithmetic for a bounded predicate. This keeps integer casts
    /// of predicates symbolic, notably the remainder term in ceiling division.
    fn indicator(&self, predicate: &Predicate) -> Option<Sym> {
        if let Some(value) = self.truth(predicate) { return Some(Sym::constant(i64::from(value))); }
        Some(match predicate {
            Predicate::Constant(value) => Sym::constant(i64::from(*value)),
            Predicate::Nonnegative(value) => {
                let (lo, hi) = self.bounds(value)?;
                // -scale <= value < scale, hence floor((value+scale)/scale)
                // is exactly zero below zero and one at or above zero.
                let scale = i64::try_from((-i128::from(lo)).max(i128::from(hi) + 1)).ok()?;
                let scale = Sym::constant(scale);
                value.add(&scale).quot(&scale)
            },
            Predicate::Not(value) => Sym::constant(1).sub(&self.indicator(value)?),
            Predicate::And(values) => {
                let mut result = Sym::constant(1);
                for value in values { result = result.mul(&self.indicator(value)?); }
                result
            },
        })
    }
    fn formula(&self, state: &Derivation<'_>, expression: &Expression, lane: usize) -> Option<Predicate> {
        use Expression as E;
        if let E::Helper(helper, arguments, element) = expression {
            return self.formula(state, &crate::terminal::helper::single_expression(*helper, arguments, *element)?, lane);
        }
        if let E::Variable(name, Type::Bool) = expression {
            if let Some(predicate) = self.predicates.get(name).and_then(|values| values[lane].clone()) { return Some(predicate); }
        }
        match expression {
            E::Cast(Type::Bool, inner) => self.formula(state, inner, lane),
            E::Unary(UnaryOp::Not, inner, _) => Some(Predicate::Not(Box::new(self.formula(state, inner, lane)?))),
            E::Binary(operation, left, right, Type::Bool)
                if matches!(operation, BinaryOp::And | BinaryOp::Or)
                    || (matches!(operation, BinaryOp::BitAnd | BinaryOp::BitOr)
                        && left.ty() == Type::Bool && right.ty() == Type::Bool) => {
                let left = self.formula(state, left, lane)?;
                let right = self.formula(state, right, lane)?;
                if matches!(operation, BinaryOp::Or | BinaryOp::BitOr) {
                    Some(Predicate::Not(Box::new(Predicate::And(vec![
                        Predicate::Not(Box::new(left)), Predicate::Not(Box::new(right)),
                    ]))))
                } else { Some(Predicate::And(vec![left, right])) }
            },
            E::ShortCircuit { or, left, right } => {
                let left = self.formula(state, left, lane)?;
                let right = self.formula(state, right, lane)?;
                if *or { Some(Predicate::Not(Box::new(Predicate::And(vec![Predicate::Not(Box::new(left)), Predicate::Not(Box::new(right))])))) }
                else { Some(Predicate::And(vec![left, right])) }
            },
            _ => {
                let comparison = match expression {
                    E::Binary(operation, left, right, _) if matches!(operation, BinaryOp::Lt | BinaryOp::Le | BinaryOp::Gt | BinaryOp::Ge | BinaryOp::Eq | BinaryOp::Ne) => (*operation, left.as_ref(), right.as_ref()),
                    _ => {
                        let value = self.expression(state, expression, lane)?;
                        if let Some(value) = value.as_constant() { return Some(Predicate::Constant(value != 0)); }
                        let zero = Predicate::And(vec![Predicate::Nonnegative(value.clone()), Predicate::Nonnegative(Sym::constant(0).sub(&value))]);
                        return Some(Predicate::Not(Box::new(zero)));
                    },
                };
                let left = self.expression(state, comparison.1, lane)?;
                let right = self.expression(state, comparison.2, lane)?;
                Some(match comparison.0 {
                    BinaryOp::Lt => Predicate::Nonnegative(right.sub(&left).sub(&Sym::constant(1))),
                    BinaryOp::Le => Predicate::Nonnegative(right.sub(&left)),
                    BinaryOp::Gt => Predicate::Nonnegative(left.sub(&right).sub(&Sym::constant(1))),
                    BinaryOp::Ge => Predicate::Nonnegative(left.sub(&right)),
                    operation => {
                        let equal = Predicate::And(vec![Predicate::Nonnegative(left.sub(&right)), Predicate::Nonnegative(right.sub(&left))]);
                        if operation == BinaryOp::Eq { equal } else { Predicate::Not(Box::new(equal)) }
                    },
                })
            },
        }
    }
    /// Only a uniform compile-time predicate can change terminal topology here.
    /// Data-dependent or varying runtime predicates retain their original path.
    pub fn predicate(&self, state: &Derivation<'_>, builder: &mut ModelBuilder, expression: &Expression) -> Result<Option<VarId>, DerivationError> {
        let participation = self.participation(state, expression);
        if (0..32).any(|lane| state.active & (1 << lane) != 0 && participation[lane].is_none()) { return Ok(None); }
        let Some(reference) = participation.iter().flatten().next() else { return Ok(None); };
        if participation.iter().flatten().any(|predicate| predicate != reference) { return Ok(None); }
        fn append(builder: &mut ModelBuilder, expressions: &mut Expressions, predicate: &Predicate) -> Result<VarId, DerivationError> {
            Ok(match predicate {
                Predicate::Constant(value) => builder.variable("metal.parameter.constant", Domain::singleton(i64::from(*value))),
                Predicate::Nonnegative(expression) => expressions.predicate(builder, "metal.parameter.predicate", expression)
                    .map_err(|error| DerivationError::Unsupported(error.to_string()))?,
                Predicate::Not(inner) => {
                    let inner = append(builder, expressions, inner)?;
                    let output = builder.variable("metal.parameter.not", Domain::boolean());
                    builder.constraint(Constraint::NotEqual {left: output, right: inner}); output
                },
                Predicate::And(inputs) => {
                    let inputs = inputs.iter().map(|input| append(builder, expressions, input)).collect::<Result<Vec<_>, _>>()?;
                    let output = builder.variable("metal.parameter.and", Domain::boolean());
                    builder.constraint(Constraint::BoolAnd {output, inputs}); output
                },
            })
        }
        let mut next = builder.clone();
        let mut expressions = Expressions::new(self.parameters.clone());
        let result = append(&mut next, &mut expressions, &reference)?;
        *builder = next; Ok(Some(result))
    }
}
