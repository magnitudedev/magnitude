//! Demand the checked reference meaning from the actual closed computation.
//!
//! These are private analysis terms, not another executable or source witness.
//! Both projections intern into one graph; only actual term equality can close
//! a relation. Unsupported structure retains a reason and remains unresolved.
use super::*;
use seismic_ir::repr::ScalarKind;
use seismic_lang::entry::SemanticProgram;
use seismic_lang::reference_math as reference;
use std::collections::HashMap;
mod packed;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum DerivedOutcome {
    Exact,
    /// Discrete behavior is established; floating result/state values have no
    /// established error bound. Only Unconstrained can consume this region.
    DiscreteProvenFloatingUnbounded,
    Pending(&'static str),
}

use super::closed_scalar::*;

#[derive(Clone, Debug, PartialEq, Eq)]
enum Value {
    Scalar(Term),
    Range(Term, Term),
    Tuple(Vec<Value>),
    Unit,
    Tensor(AnyBufferView),
}
impl Value {
    fn flatten(&self, terms: &mut Vec<Term>) -> Result<()> {
        match self {
            Self::Scalar(value) => terms.push(*value),
            Self::Range(start,end) => terms.extend([*start,*end]),
            Self::Tuple(values) => for value in values { value.flatten(terms)?; },
            Self::Unit => (),
            Self::Tensor(_) => return Err("tensor recurrence relation is unfinished"),
        }
        Ok(())
    }
    fn replace(&self, terms: &mut impl Iterator<Item=Term>) -> Result<Self> {
        let mut next = || terms.next().ok_or("scalar product arity differs");
        Ok(match self {
            Self::Scalar(_) => Self::Scalar(next()?),
            Self::Range(..) => Self::Range(next()?,next()?),
            Self::Tuple(values) => Self::Tuple(values.iter().map(|value|value.replace(terms)).collect::<Result<_>>()?),
            Self::Unit => Self::Unit,
            Self::Tensor(_) => return Err("tensor recurrence relation is unfinished"),
        })
    }
    fn scalar(&self) -> Result<Term> {
        match self {
            Self::Scalar(value) => Ok(*value),
            _ => Err("non-scalar value in scalar relation"),
        }
    }
}

impl Analysis<'_> {
    fn input(&mut self, value: Bound) -> Result<Value> {
        Ok(match value {
            Bound::Scalar(ScalarBinding::Value { symbol, dtype }) => {
                let value = self
                    .terms
                    .node(Node::Input(symbol, ScalarKind::Scalar(dtype)));
                self.inputs.insert(symbol, value);
                Value::Scalar(value)
            }
            Bound::Scalar(ScalarBinding::Published(slot)) => {
                let value = self.terms.node(Node::Input(slot.symbol(), slot.kind()));
                self.inputs.insert(slot.symbol(), value);
                Value::Scalar(value)
            }
            Bound::Scalar(ScalarBinding::Index(expression)) => {
                if let seismic_lang::expr::NodeView::Symbol(symbol) =
                    self.expressions.view(expression.into())
                {
                    let value = self.terms.node(Node::Input(symbol, ScalarKind::Nat64));
                    self.inputs.insert(symbol, value);
                }
                Value::Scalar(self.expression(expression.into(), &HashMap::new())?)
            }
            Bound::Scalar(ScalarBinding::Integer(expression)) => {
                Value::Scalar(self.expression(expression.into(), &HashMap::new())?)
            }
            Bound::Scalar(ScalarBinding::Quantity(_)) => {
                return Err("exact host quantity comparison is not an alternative-analysis rule")
            }
            Bound::Range { start, end } => {
                Value::Range(self.input(*start)?.scalar()?, self.input(*end)?.scalar()?)
            }
            Bound::Tuple(values) => Value::Tuple(
                values
                    .into_iter()
                    .map(|v| self.input(v))
                    .collect::<Result<_>>()?,
            ),
            Bound::Unit => Value::Unit,
            Bound::Tensor(TensorRealization::Stored(value)) => {
                let root = self.storage_root(value.root)?;
                self.external.insert(root);
                Value::Tensor(*value.view.direct_backing().ok_or("mapped input relation is unfinished")?)
            }
            Bound::Tensor(TensorRealization::Computed(_)) => {
                return Err("deferred tensor input relation is unfinished")
            }
        })
    }

    fn source(
        &mut self,
        program: &SemanticProgram,
        function: &SemanticFunction,
        inputs: &[Value],
        state: &mut State,
    ) -> Result<Vec<Value>> {
        let mut values = function
            .parameters()
            .iter()
            .zip(inputs)
            .map(|(formal, value)| (formal.value, value.clone()))
            .collect::<HashMap<_, _>>();
        self.source_region(program, function, function.root(), &mut values, state)?;
        function.results().iter().map(|result| values.get(result).cloned().ok_or("source result unavailable")).collect()
    }

    fn source_region(
        &mut self,
        program: &SemanticProgram,
        function: &SemanticFunction,
        region: RegionId,
        values: &mut HashMap<SemanticValueId, Value>,
        state: &mut State,
    ) -> Result<()> {
        for (node_id, node) in function.nodes(region) {
            match node.view() {
                SemanticNodeView::Primitive {
                    primitive,
                    inputs,
                    output,
                } => {
                    let operands = inputs
                        .iter()
                        .map(|input| {
                            values
                                .get(input)
                                .cloned()
                                .ok_or("source operand unavailable")
                        })
                        .collect::<Result<Vec<_>>>()?;
                    let value = match primitive {
                        PrimitiveId::Constant(value) => Value::Scalar(self.terms.scalar(*value)),
                        PrimitiveId::Symbolic(expression) => {
                            // Symbolic is an actual checked quantity operation. Its
                            // operands are bound through the expression owner's
                            // RuntimeValue symbols, not inferred from output bounds.
                            let mut operands = state.slots.clone();
                            for symbol in self.expressions.free_symbols((*expression).into()) {
                                if let SymbolKind::RuntimeValue(value) = self.expressions.symbol_kind(symbol) {
                                    operands.insert(symbol, values.get(&value).ok_or("source quantity operand unavailable")?.scalar()?);
                                }
                            }
                            Value::Scalar(self.expression((*expression).into(), &operands)?)
                        }

                        PrimitiveId::Select => Value::Scalar(self.terms.select(
                            operands[0].scalar()?,
                            operands[1].scalar()?,
                            operands[2].scalar()?,
                        )),
                        PrimitiveId::RangeMake => {
                            Value::Range(operands[0].scalar()?, operands[1].scalar()?)
                        }
                        PrimitiveId::RangeStart => {
                            let Value::Range(start, _) = operands[0] else {
                                return Err("source range start kind");
                            };
                            Value::Scalar(start)
                        }
                        PrimitiveId::RangeEnd => {
                            let Value::Range(_, end) = operands[0] else {
                                return Err("source range end kind");
                            };
                            Value::Scalar(end)
                        }
                        _ => {
                            let recipe = function
                                .scalar_recipe(primitive, inputs)
                                .ok_or("source primitive relation is unfinished")?;
                            let operands = operands
                                .iter()
                                .map(Value::scalar)
                                .collect::<Result<Vec<_>>>()?;
                            let (value, failures) = self.terms.recipe(&recipe, &operands);
                            for (failed, cause) in failures {
                                self.failure(
                                    state,
                                    failed,
                                    SourceFailure::at(
                                        function,
                                        node_id,
                                        SourceFailureCause::Scalar(cause),
                                    ),
                                );
                            }
                            Value::Scalar(value)
                        }
                    };
                    values.insert(output, value);
                }
                SemanticNodeView::Call {
                    family,
                    inputs,
                    outputs,
                } => {
                    let inputs = inputs
                        .iter()
                        .map(|input| {
                            values
                                .get(input)
                                .cloned()
                                .ok_or("source call operand unavailable")
                        })
                        .collect::<Result<Vec<_>>>()?;
                    let reference = program.function(program.family(family).reference().function());
                    let result = self.source(program, reference, &inputs, state)?;
                    for (output, value) in outputs.iter().zip(result) {
                        values.insert(*output, value);
                    }
                }
                SemanticNodeView::TuplePack { inputs, output } => {
                    let operands = inputs
                        .iter()
                        .map(|input| {
                            values
                                .get(input)
                                .cloned()
                                .ok_or("source tuple operand unavailable")
                        })
                        .collect::<Result<_>>()?;
                    values.insert(output, Value::Tuple(operands));
                }
                SemanticNodeView::TupleGet {
                    tuple,
                    index,
                    output,
                } => {
                    let Some(Value::Tuple(tuple)) = values.get(&tuple) else {
                        return Err("source tuple kind");
                    };
                    let value = tuple[index as usize].clone();
                    values.insert(output, value);
                }
                SemanticNodeView::Extent {
                    tensor,
                    axis,
                    output,
                } => {
                    let Some(Value::Tensor(view)) = values.get(&tensor) else {
                        return Err("source extent has no stored view");
                    };
                    let extent = self.storage.view(*view).extents[axis as usize];
                    let term = self.expression(extent.into(), &state.slots)?;
                    let term = match function.value(output).ty {
                        SemanticType::Scalar(dtype @ (DType::I32|DType::U32)) => self.terms.word_from_natural(term,dtype),
                        SemanticType::Index{..} => term,
                        _ => return Err("source extent has no established scalar meaning"),
                    };
                    values.insert(output, Value::Scalar(term));
                }
                SemanticNodeView::ElementRead {
                    place,
                    indices,
                    output,
                } => {
                    let Some(Value::Tensor(view)) = values.get(&place) else {
                        return Err("source read has no stored place");
                    };
                    let coordinates = indices
                        .iter()
                        .map(|index| self.source_coordinate(function, values, *index))
                        .collect::<Result<Vec<_>>>()?;
                    let value = self.source_tensor_read(*view, &coordinates, state)?;
                    values.insert(output, Value::Scalar(value));
                }
                SemanticNodeView::ElementWrite {
                    place,
                    indices,
                    value,
                    output,
                } => {
                    let Some(Value::Tensor(view)) = values.get(&place) else {
                        return Err("source write has no stored place");
                    };
                    let view = *view;
                    let coordinates = indices
                        .iter()
                        .map(|index| self.source_coordinate(function, values, *index))
                        .collect::<Result<Vec<_>>>()?;
                    let place = self.place(view, &coordinates, state)?;
                    self.write(state, place, values[&value].scalar()?);
                    values.insert(output, Value::Tensor(view));
                }
                SemanticNodeView::Loop { kind, start, end, captures, body, carries, .. }
                    if kind == seismic_lang::entry::LoopKind::Parallel =>
                {
                    if !carries.is_empty() {
                        return Err("checked parallel loops carry no products");
                    }
                    let start = self.source_index(function, start, values)?;
                    let end = self.source_index(function, end, values)?;
                    let extent = match (&self.terms.nodes[start.0], &self.terms.nodes[end.0]) {
                        (Node::Natural(0), _) => end,
                        (Node::Natural(a), Node::Natural(b)) => self.terms.node(Node::Natural(b.saturating_sub(*a))),
                        _ => return Err("offset parallel map relation is unfinished"),
                    };
                    let body_region = function.region(body);
                    let (binder, parameters) = body_region.parameters().split_first().ok_or("source loop binder unavailable")?;
                    if parameters.len() != captures.len() { return Err("source loop capture arity differs"); }
                    let mut iteration = values.clone();
                    for (parameter, capture) in parameters.iter().zip(captures) {
                        iteration.insert(*parameter, values.get(capture).cloned().ok_or("source loop capture unavailable")?);
                    }
                    let binder = *binder;
                    self.map_visit(state, extent, &mut |analysis, visit, lane| {
                        let index = analysis.terms.natural_binary(false, start, visit);
                        let mut iteration = iteration.clone();
                        iteration.insert(binder, Value::Scalar(index));
                        analysis.source_region(program, function, body, &mut iteration, lane)
                    })?;
                }
                SemanticNodeView::Loop { start, end, captures, body, carries, .. } => {
                    let start = self.source_index(function, start, values)?;
                    let end = self.source_index(function, end, values)?;
                    let depth = self.loop_depth;
                    let body_region = function.region(body);
                    let mut iteration = values.clone();
                    let (binder, parameters) = body_region.parameters().split_first().ok_or("source loop binder unavailable")?;
                    if parameters.len() != captures.len() { return Err("source loop capture arity differs"); }
                    for (parameter, capture) in parameters.iter().zip(captures) {
                        iteration.insert(*parameter, values.get(capture).cloned().ok_or("source loop capture unavailable")?);
                    }
                    iteration.insert(*binder, Value::Scalar(self.terms.node(Node::Iteration(depth))));
                    let mut initial = Vec::new();
                    let mut schemas = Vec::new();
                    for carry in carries {
                        let value = values.get(&carry.initial).ok_or("source initial carry unavailable")?;
                        let ty = &function.value(carry.parameter).ty;
                        let header = self.source_header(value, ty, depth, &mut initial)?;
                        schemas.push(header.clone());
                        iteration.insert(carry.parameter, header);
                    }
                    let mut body_state = state.clone();
                    self.loop_depth += 1;
                    let body_result = self.source_region(program, function, body, &mut iteration, &mut body_state);
                    self.loop_depth = depth;
                    body_result?;
                    if body_state.writes.len() != state.writes.len() || body_state.effects.len() != state.effects.len() {
                        return Err("source stateful recurrence relation is unfinished");
                    }
                    let mut next = Vec::new();
                    for carry in carries {
                        iteration.get(&carry.yielded).ok_or("source next carry unavailable")?.flatten(&mut next)?;
                    }
                    let mut folded = self.terms.fold(start, end, initial, next).into_iter();
                    for (carry, schema) in carries.iter().zip(schemas) {
                        values.insert(carry.result, schema.replace(&mut folded)?);
                    }
                    if folded.next().is_some() { return Err("source carry result arity differs"); }
                }
                SemanticNodeView::Check { condition, reason } => {
                    let failed = self.terms.not(values[&condition].scalar()?);
                    self.failure(
                        state,
                        failed,
                        SourceFailure::at(
                            function,
                            node_id,
                            SourceFailureCause::Check(reason.clone()),
                        ),
                    );
                }
                _ => return Err("source state or control relation is unfinished"),
            }
        }
        Ok(())
    }

    /// One checked element coordinate as a natural term.
    fn source_coordinate(&mut self, function: &SemanticFunction, values: &HashMap<SemanticValueId, Value>, index: SemanticValueId) -> Result<Term> {
        let value = values.get(&index).ok_or("source coordinate unavailable")?.scalar()?;
        match function.value(index).ty {
            SemanticType::Scalar(dtype) => self.terms.natural(value, dtype),
            SemanticType::Index { .. } => self.terms.natural(value, DType::U32),
            SemanticType::Integer => self.terms.exact_natural(value),
            _ => Err("source index kind"),
        }
    }

    fn source_index(&mut self, function: &SemanticFunction, value: SemanticValueId, values: &HashMap<SemanticValueId, Value>) -> Result<Term> {
        let term = values.get(&value).ok_or("source index unavailable")?.scalar()?;
        match function.value(value).ty {
            SemanticType::Index { .. } => Ok(term),
            SemanticType::Scalar(dtype @ (DType::U32 | DType::I32)) => self.terms.natural(term, dtype),
            _ => Err("source mathematical index relation is unfinished"),
        }
    }

    fn source_header(&mut self, value: &Value, ty: &SemanticType, depth: u32, initial: &mut Vec<Term>) -> Result<Value> {
        Ok(match (value, ty) {
            (Value::Scalar(value), SemanticType::Scalar(dtype)) => {
                let ordinal = initial.len() as u32;
                initial.push(*value);
                Value::Scalar(self.terms.node(Node::Header { depth, ordinal, kind: ScalarKind::Scalar(*dtype) }))
            }
            (Value::Scalar(value), SemanticType::Index { .. }) => {
                let ordinal = initial.len() as u32;
                initial.push(*value);
                Value::Scalar(self.terms.node(Node::Header { depth, ordinal, kind: ScalarKind::Nat64 }))
            }
            (Value::Tuple(values), SemanticType::Tuple(types)) if values.len() == types.len() => Value::Tuple(values.iter().zip(types).map(|(v,t)|self.source_header(v,t,depth,initial)).collect::<Result<_>>()?),
            (Value::Unit, SemanticType::Void) => Value::Unit,
            (Value::Range(start,end), SemanticType::Range { .. }) => {
                let ordinal = initial.len() as u32;
                initial.extend([*start,*end]);
                Value::Range(self.terms.node(Node::Header { depth, ordinal, kind: ScalarKind::Nat64 }),self.terms.node(Node::Header { depth, ordinal:ordinal+1, kind: ScalarKind::Nat64 }))
            },
            _ => return Err("source carry product relation is unfinished"),
        })
    }

    fn actual(
        &mut self,
        bindings: &BindingArena,
        id: BindingId,
        slots: &HashMap<seismic_lang::expr::SymbolId, Term>,
    ) -> Result<Value> {
        let id = bindings.selected(id, &bindings.parameter_selections);
        fn value(
            analysis: &mut Analysis<'_>,
            bound: Bound,
            slots: &HashMap<seismic_lang::expr::SymbolId, Term>,
        ) -> Result<Value> {
            Ok(match bound {
                Bound::Scalar(ScalarBinding::Published(slot)) => Value::Scalar(
                    *slots
                        .get(&slot.symbol())
                        .ok_or("physical result slot was not written")?,
                ),
                Bound::Scalar(ScalarBinding::Value { symbol, .. }) => Value::Scalar(
                    *slots
                        .get(&symbol)
                        .or_else(|| analysis.inputs.get(&symbol))
                        .ok_or("physical result symbol unavailable")?,
                ),
                Bound::Scalar(ScalarBinding::Index(expression)) => {
                    Value::Scalar(analysis.expression(expression.into(), slots)?)
                }
                Bound::Scalar(ScalarBinding::Integer(expression)) => {
                    Value::Scalar(analysis.expression(expression.into(), slots)?)
                }
                Bound::Scalar(ScalarBinding::Quantity(_)) => {
                    return Err("exact host quantity comparison is not an alternative-analysis rule")
                }
                Bound::Range { start, end } => Value::Range(
                    value(analysis, *start, slots)?.scalar()?,
                    value(analysis, *end, slots)?.scalar()?,
                ),
                Bound::Tuple(values) => Value::Tuple(
                    values
                        .into_iter()
                        .map(|v| value(analysis, v, slots))
                        .collect::<Result<_>>()?,
                ),
                Bound::Unit => Value::Unit,
                Bound::Tensor(_) => return Err("physical tensor result relation is unfinished"),
            })
        }
        value(self, bindings.get(id), slots)
    }
}

pub(crate) fn derive<B: seismic_native_target::TargetFamily>(
    arena: &ExprArena,
    program: &SemanticProgram,
    function: &SemanticFunction,
    bindings: &BindingArena,
    executable: &seismic_ir::execution::ClosedExecutableIr<B>,
) -> DerivedOutcome {
    let mut analysis = Analysis::new(arena, executable.storage(), executable.schedule().steps());
    let result = (|| -> Result<DerivedOutcome> {
        let family = program
            .families()
            .find(|(_, family)| {
                family
                    .candidates()
                    .iter()
                    .any(|candidate| candidate.function == function.id())
            })
            .ok_or("constructed body has no checked family")?
            .1;
        let reference = program.function(family.reference().function());
        let inputs = bindings
            .parameters
            .iter()
            .map(|id| {
                analysis.input(bindings.get(bindings.selected(*id, &bindings.parameter_selections)))
            })
            .collect::<Result<Vec<_>>>()?;
        let mut source = State::default();
        let expected = analysis.source(program, reference, &inputs, &mut source)?;
        let mut physical = State::default();
        analysis.schedule(
            executable.schedule(),
            executable.kernels(),
            executable.schedule().steps(),
            &mut physical,
        )?;
        let mut actual = bindings
            .results
            .iter()
            .map(|id| analysis.actual(bindings, *id, &physical.slots))
            .collect::<Result<Vec<_>>>()?;
        let mut floating_difference = false;
        let mut successful = Vec::new();
        for effect in source.effects.iter().chain(&physical.effects) {
            let observed = match effect {
                Effect::Write(place, value) => vec![(Some(place), *value)],
                Effect::Failure(value, _) => vec![(None, *value)],
                Effect::Map { extent, writes } => std::iter::once((None, *extent))
                    .chain(writes.iter().map(|(place, value)| (Some(place), *value))).collect(),
            };
            for (place, value) in observed {
                if place.is_some_and(|place| analysis.terms.contains_opaque(place.byte)) { return Err("opaque observed write address"); }
                if analysis.terms.contains_opaque(value) { return Err("opaque observed state or failure value"); }
            }
        }
        if source.effects.len() != physical.effects.len() {
            return Err("ordered observable effects differ from reference");
        }
        for (expected, actual) in source.effects.iter().zip(&physical.effects) {
            match (expected, actual) {
                (
                    Effect::Failure(expected, expected_cause),
                    Effect::Failure(actual, actual_cause),
                ) if analysis.terms.under(*expected, &successful)
                    == analysis.terms.under(*actual, &successful)
                    && expected_cause == actual_cause =>
                {
                    successful.push((*expected, false));
                    successful.push((*actual, false));
                }
                (Effect::Map { extent: expected_extent, writes: expected }, Effect::Map { extent: actual_extent, writes: actual })
                    if expected_extent == actual_extent && expected.len() == actual.len()
                        && expected.iter().zip(actual).all(|((a, _), (b, _))| a == b) =>
                {
                    for ((place, expected_value), (_, actual_value)) in expected.iter().zip(actual) {
                        if analysis.terms.under(*expected_value, &successful)
                            != analysis.terms.under(*actual_value, &successful)
                        {
                            if !matches!(place.dtype, DType::F16 | DType::BF16 | DType::F32) {
                                return Err("discrete writable-state equality is not established");
                            }
                            floating_difference = true;
                        }
                    }
                }
                (Effect::Write(expected, expected_value), Effect::Write(actual, actual_value))
                    if expected == actual =>
                {
                    if analysis.terms.under(*expected_value, &successful)
                        != analysis.terms.under(*actual_value, &successful)
                    {
                        if !matches!(expected.dtype, DType::F16 | DType::BF16 | DType::F32) {
                            return Err("discrete writable-state equality is not established");
                        }
                        floating_difference = true;
                    }
                }
                _ => {
                    return Err(
                        "write address/order or failure-prefix equivalence is not established",
                    )
                }
            }
        }
        fn restrict(value: &mut Value, terms: &mut Terms, successful: &[(Term, bool)]) {
            match value {
                Value::Scalar(value) => *value = terms.under(*value, successful),
                Value::Range(start, end) => {
                    *start = terms.under(*start, successful);
                    *end = terms.under(*end, successful);
                }
                Value::Tuple(values) => {
                    for value in values {
                        restrict(value, terms, successful);
                    }
                }
                Value::Unit | Value::Tensor(_) => {}
            }
        }
        let mut expected = expected;
        for value in expected.iter_mut().chain(&mut actual) {
            restrict(value, &mut analysis.terms, &successful);
        }
        fn opaque(value:&Value,terms:&Terms) -> bool {
            match value {
                Value::Scalar(value) => terms.contains_opaque(*value),
                Value::Range(a,b) => terms.contains_opaque(*a)||terms.contains_opaque(*b),
                Value::Tuple(values) => values.iter().any(|value|opaque(value,terms)),
                _ => false,
            }
        }
        if actual.iter().chain(&expected).any(|value|opaque(value,&analysis.terms)) { return Err("opaque returned value"); }
        if actual == expected && !floating_difference {
            return Ok(DerivedOutcome::Exact);
        }
        if actual.len() != expected.len() {
            return Err("actual result arity differs from checked reference");
        }
        for ((actual, expected), result) in actual.iter().zip(&expected).zip(reference.results()) {
            if actual == expected {
                continue;
            }
            if !matches!(
                &reference.value(*result).ty,
                SemanticType::Scalar(DType::F16 | DType::BF16 | DType::F32)
            ) {
                return Err("discrete result equality is not established");
            }
        }
        Ok(DerivedOutcome::DiscreteProvenFloatingUnbounded)
    })();
    result.unwrap_or_else(DerivedOutcome::Pending)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::candidate_domain::{
        construct_candidate_domain, BodyMapping, ConstructionAllowance, ConstructionCoordinate,
        Materialization,
    };
    use crate::realization::demand_driven_tests::registry;
    use seismic_lang::checked::{check_source, SourceFile, SourceSet};

    fn analyze(source: &str, authored: bool) -> DerivedOutcome {
        analyze_mapped(source, authored.then_some(BodyMapping::Authored))
    }

    /// The universal member, or the root body constructed with `mapping`.
    fn analyze_mapped(source: &str, mapping: Option<BodyMapping>) -> DerivedOutcome {
        let module = check_source(SourceSet::new(vec![SourceFile {
            path: "outcome-relation.seismic".into(),
            text: source.into(),
        }]))
        .unwrap();
        let entry = module
            .entry(module.entry_named("probe").unwrap(), &Default::default())
            .unwrap();
        let device = super::super::import_tests::device();
        let registry = registry();
        let mut domain = construct_candidate_domain(
            entry,
            &device,
            &registry,
            &seismic_lang::precision::PrecisionPolicy::Exact,
        )
        .unwrap();
        if let Some(mapping) = mapping {
            let selected = domain
                .root_selections()
                .into_iter()
                .find(|selected| selected.mapping == mapping)
                .unwrap();
            let state = domain
                .advance(
                    &ConstructionCoordinate::root(selected),
                    ConstructionAllowance {
                        work_units: 100_000,
                        wall_time: std::time::Duration::from_secs(30),
                    },
                )
                .state;
            assert!(matches!(state, Materialization::Ready(_)), "{state:?}");
        }
        let parts = domain.into_parts();
        let candidate = if mapping.is_some() {
            &parts.materialized.as_slice().last().unwrap().family
        } else {
            &parts.materialized.first().family
        };
        let function = parts
            .source_program
            .functions()
            .find(|(_, function)| function.stable() == candidate.provenance().root)
            .unwrap()
            .1;
        derive(
            &parts.arena,
            &parts.source_program,
            function,
            &candidate.bindings().arena,
            candidate.executable(),
        )
    }

    #[test]
    fn independent_participants_relate_to_the_parallel_map_visits() {
        let map = "fn probe(input: &tensor[4] f32, out: &mut tensor[4] f32):\n    parallel for i in 0..4:\n        out[i] = input[i] + 1.0\n";
        assert_eq!(analyze_mapped(map, Some(BodyMapping::Independent)), DerivedOutcome::Exact);
        assert_eq!(analyze_mapped(map, None), DerivedOutcome::Exact);
        let symbolic = "fn probe[N](input: &tensor[N] f32, out: &mut tensor[N] f32):\n    parallel for i in 0..N:\n        out[i] = input[i] * 2.0\n";
        assert_eq!(analyze_mapped(symbolic, Some(BodyMapping::Independent)), DerivedOutcome::Exact);
        // A later read of mapped storage stays unfinished, and so does a
        // different authored participant map.
        let later = "fn probe(input: &tensor[4] f32, out: &mut tensor[4] f32) -> f32:\n    parallel for i in 0..4:\n        out[i] = input[i]\n    return out[0]\n";
        assert!(matches!(analyze_mapped(later, Some(BodyMapping::Independent)), DerivedOutcome::Pending(_)));
        let authored = format!("{map}\nlower probe(input: &tensor[4] f32, out: &mut tensor[4] f32) for cpu:\n    parallel for i in 0..4:\n        out[i] = input[i] + 2.0\n");
        assert_ne!(analyze_mapped(&authored, Some(BodyMapping::Authored)), DerivedOutcome::Exact);
    }

    #[test]
    fn actual_closed_scalar_recipes_match_the_checked_reference_graph() {
        for source in [
            "fn probe(a: f32) -> f32:\n    return a\n",
            "fn probe(a: f32, b: f32) -> f32:\n    return a + b\n",
            "fn probe(a: bf16, b: bf16, c: bf16) -> bf16:\n    return fma(a, b, c)\n",
            "fn probe(a: f32, b: f32) -> f32:\n    return max(a, b)\n",
            "fn plus(a: f32, b: f32) -> f32:\n    return a + b\n\nfn probe(a: f32, b: f32) -> f32:\n    return plus(a, b)\n",
        ] {
            assert_eq!(analyze(source, false), DerivedOutcome::Exact, "{source}");
        }
    }

    #[test]
    fn ordered_scalar_recurrence_uses_a_symbolic_header_and_actual_backedge() {
        assert_eq!(analyze("fn probe(a: f32, b: f32) -> f32:\n    let mut total = a\n    for i in 0..3:\n        total = fma(total, b, a)\n    return total\n", false), DerivedOutcome::Exact);
    }

    #[test]
    fn q4g64_fields_and_bf16_activations_follow_ordered_f32_fma_reference() {
        assert_eq!(analyze("fn probe(a: &tensor[128] bf16, w: &tensor[128] q4g64) -> f32:\n    let first = fma(f32(a[63]), w[63], 0.0)\n    return fma(f32(a[64]), w[64], first)\n",false), DerivedOutcome::Exact);
    }

    #[test]
    fn authored_labels_neither_establish_nor_prevent_exact_scalar_equivalence() {
        assert_eq!(analyze("fn probe(a: f32, b: f32) -> f32:\n    return a + b\n\nlower probe(a: f32, b: f32) -> f32 for cpu:\n    return a + b\n", true), DerivedOutcome::Exact);
        assert_eq!(analyze("fn probe(a: f32, b: f32) -> f32:\n    return a + b\n\nlower probe(a: f32, b: f32) -> f32 for cpu:\n    return a - b\n", true), DerivedOutcome::DiscreteProvenFloatingUnbounded);
        assert!(matches!(analyze("fn probe(a: u32) -> u32:\n    return a\n\nlower probe(a: u32) -> u32 for cpu:\n    return 3\n", true), DerivedOutcome::Pending(_)));
    }

    #[test]
    fn unit_helpers_compare_actual_ordered_writes_including_addresses() {
        assert_eq!(analyze("fn write(out: &mut tensor[2] f32, value: f32):\n    out[0] = value\n\nfn probe(out: &mut tensor[2] f32):\n    write(out, 3.0)\n    write(out, 7.0)\n", false), DerivedOutcome::Exact);
        assert_eq!(analyze("fn probe(out: &mut tensor[2] f32):\n    out[0] = 3.0\n\nlower probe(out: &mut tensor[2] f32) for cpu:\n    out[0] = 7.0\n", true), DerivedOutcome::DiscreteProvenFloatingUnbounded);
        assert!(matches!(analyze("fn probe(out: &mut tensor[2] f32):\n    out[0] = 3.0\n\nlower probe(out: &mut tensor[2] f32) for cpu:\n    out[1] = 3.0\n", true), DerivedOutcome::Pending(_)));
    }

    #[test]
    fn actual_failure_continuation_retains_prior_writes_and_source_cause_order() {
        let source = "fn probe(out: &mut tensor[1] f32, numerator: i32, divisor: i32) -> i32:\n    out[0] = 7.0\n    let value = numerator / divisor\n    out[0] = 9.0\n    return value\n";
        assert_eq!(analyze(source, false), DerivedOutcome::Exact);
        let altered = format!("{source}\nlower probe(out: &mut tensor[1] f32, numerator: i32, divisor: i32) -> i32 for cpu:\n    let value = numerator / divisor\n    out[0] = 7.0\n    out[0] = 9.0\n    return value\n");
        assert!(matches!(
            analyze(&altered, true),
            DerivedOutcome::Pending(_)
        ));
        assert_eq!(analyze("fn write(out: &mut tensor[1] f32, divisor: i32):\n    out[0] = 7.0\n    let result = 1 / divisor\n    out[0] = f32(result)\n\nfn probe(out: &mut tensor[1] f32, divisor: i32):\n    write(out, divisor)\n",false),DerivedOutcome::Exact,"effect-only imported call preserves actual prior writes and failure");
    }
}
