use super::scalar;
use super::tensor::{read_bits, write_bits};
use super::value::{row_major, Backing, Scalar, TensorValue, Value};
use super::{Arg, Interpreter, TensorData};
use crate::entry::{
    Candidate, CheckReason, LoopKind, NumericalRole, ParameterKind, RegionKind, ScalarRef,
    SemanticFunction, SemanticNodeView, SemanticType, TensorStorage, ViewTransform,
};
use crate::expr::{
    compiled::InvocationValues, Assignment, PartialAssignment, SymbolKind, SymbolValue,
};
use crate::ids::{FamilyId, RepresentationConversionId, SemanticValueId};
use crate::intrinsics::{accumulator_dtype, AtomicOp, Constant, PrimitiveId, ReduceOp};
use crate::registry::{self, PlaneRepackRecipe, RepackExpr, RepresentationKind};
use crate::types::DType;
use std::collections::BTreeMap;

type Environment = BTreeMap<SemanticValueId, Value>;

impl Interpreter<'_> {
    pub(super) fn run_reference(&mut self, arguments: &[Arg]) -> Result<Vec<Value>, String> {
        if arguments.len() != self.entry.schema().parameters().len() {
            return Err(format!(
                "entry expects {} flattened arguments, got {}",
                self.entry.schema().parameters().len(),
                arguments.len()
            ));
        }
        let mut assignment = Assignment::new();
        let mut values = Vec::with_capacity(arguments.len());
        for (parameter, argument) in self.entry.schema().parameters().iter().zip(arguments) {
            let value = match (&parameter.kind, argument) {
                (ParameterKind::Tensor { representation, .. }, Arg::Tensor(index)) => {
                    let tensor = self
                        .tensors
                        .get(*index)
                        .ok_or("tensor argument index is outside the oracle table")?;
                    if tensor.representation() != *representation {
                        return Err(format!(
                            "tensor argument `{}` has representation `{}`, expected `{}`",
                            parameter.name,
                            registry::representation_info(tensor.representation()).name,
                            registry::representation_info(*representation).name
                        ));
                    }
                    Value::Tensor(TensorValue::argument(
                        *index,
                        *representation,
                        tensor.shape(),
                    ))
                }
                (ParameterKind::Scalar { dtype, symbol }, Arg::Scalar(actual, value)) => {
                    if dtype != actual {
                        return Err(format!(
                            "scalar argument `{}` has the wrong dtype",
                            parameter.name
                        ));
                    }
                    assignment.bind(*symbol, scalar_symbol(*dtype, *value));
                    Value::Scalar(*dtype, super::round_to(*dtype, *value))
                }
                (ParameterKind::Index { symbol, .. }, Arg::Index(value)) => {
                    assignment.bind(*symbol, SymbolValue::Int(*value));
                    Value::Scalar(DType::I32, *value as f64)
                }
                (ParameterKind::Range { start, end, .. }, Arg::Range(first, last)) => {
                    assignment.bind(*start, SymbolValue::Int(*first));
                    assignment.bind(*end, SymbolValue::Int(*last));
                    Value::Range(*first, *last)
                }
                _ => return Err(format!("argument kind mismatch for `{}`", parameter.name)),
            };
            values.push(value);
        }
        let mut observations = Vec::new();
        for (parameter, argument) in self.entry.schema().parameters().iter().zip(arguments) {
            let ParameterKind::Tensor { axes, .. } = &parameter.kind else {
                continue;
            };
            let Arg::Tensor(index) = argument else {
                unreachable!("checked tensor parameter is not a tensor oracle argument")
            };
            let shape = self
                .tensors
                .get(*index)
                .ok_or("tensor argument index is outside the oracle table")?
                .shape();
            if shape.len() != axes.len() {
                return Err(format!(
                    "tensor argument `{}` has the wrong rank",
                    parameter.name
                ));
            }
            observations.extend(shape.iter().map(|extent| *extent as u64));
        }
        let inference = self
            .entry
            .schema()
            .compile_dimension_inference(self.entry.arena(), &PartialAssignment::new());
        let mut inferred = InvocationValues::new();
        inference
            .infer(&observations, &mut inferred)
            .map_err(|failure| {
                format!(
                    "tensor-axis observation {} does not admit the entry's exact dimension solution",
                    failure.observation()
                )
            })?;
        for dimension in self.entry.schema().dimensions() {
            let value = inferred
                .get(dimension.symbol)
                .unwrap_or_else(|| unreachable!("sealed inference plan omitted a call dimension"));
            assignment.bind(dimension.symbol, value);
        }
        if !self
            .entry
            .arena()
            .eval_bool(self.entry.domain().predicate().node(), &assignment)
            .map_err(|error| format!("entry-domain evaluation failed: {error:?}"))?
        {
            return Err("invocation is outside the entry domain".to_owned());
        }

        let root = self.entry.program().root();
        let results = self.call_reference(root, values, &mut assignment)?;
        if results.len() != self.entry.schema().results().len() {
            unreachable!("checked root reference result arity differs from its call schema")
        }
        Ok(results)
    }

    fn reference(
        entry: &crate::entry::LogicalEntry,
        family: FamilyId,
    ) -> (&Candidate, &SemanticFunction) {
        let candidate = entry
            .program()
            .family(family)
            .candidates()
            .iter()
            .find(|candidate| candidate.numerical == NumericalRole::Reference)
            .unwrap_or_else(|| unreachable!("checked family has no reference candidate"));
        (candidate, entry.program().function(candidate.function))
    }

    fn call_reference(
        &mut self,
        family: FamilyId,
        arguments: Vec<Value>,
        assignment: &mut Assignment,
    ) -> Result<Vec<Value>, String> {
        // Copy the external entry reference before mutably borrowing the
        // oracle's tensor table; semantic arenas are immutable throughout an
        // execution.
        let entry = self.entry;
        let (_, function) = Self::reference(entry, family);
        if function.parameters().len() != arguments.len() {
            unreachable!("checked call arity differs from its family contract")
        }
        let mut environment = Environment::new();
        for (parameter, value) in function.parameters().iter().zip(arguments) {
            environment.insert(parameter.value, value);
        }
        self.execute_region(function, function.root(), &mut environment, assignment)?;
        function
            .results()
            .iter()
            .map(|result| {
                environment
                    .get(result)
                    .cloned()
                    .ok_or_else(|| "reference body did not define a declared result".to_owned())
            })
            .collect()
    }

    fn execute_region(
        &mut self,
        function: &SemanticFunction,
        region: crate::ids::RegionId,
        environment: &mut Environment,
        assignment: &mut Assignment,
    ) -> Result<(), String> {
        for (_, node) in function.nodes(region) {
            match node.view() {
                SemanticNodeView::Primitive {
                    primitive,
                    inputs,
                    output,
                } => {
                    let result = self.eval_primitive(
                        function,
                        primitive,
                        inputs,
                        output,
                        environment,
                        assignment,
                    )?;
                    environment.insert(output, result);
                }
                SemanticNodeView::Elementwise {
                    primitive,
                    inputs,
                    output,
                } => {
                    let result = self.eval_elementwise(
                        function,
                        primitive,
                        inputs,
                        output,
                        environment,
                        assignment,
                    )?;
                    environment.insert(output, result);
                }
                SemanticNodeView::Reduce {
                    op,
                    axis,
                    input,
                    output,
                    ..
                } => {
                    let result = self.eval_reduce(
                        function,
                        op,
                        axis,
                        input,
                        output,
                        environment,
                        assignment,
                    )?;
                    environment.insert(output, result);
                }
                SemanticNodeView::Call {
                    family,
                    inputs,
                    outputs,
                } => {
                    let arguments = inputs
                        .iter()
                        .map(|input| self.value(environment, *input).cloned())
                        .collect::<Result<Vec<_>, _>>()?;
                    let results = self.call_reference(family, arguments, assignment)?;
                    if results.len() != outputs.len() {
                        unreachable!("checked semantic call result arity mismatch")
                    }
                    for (output, value) in outputs.iter().zip(results) {
                        environment.insert(*output, value);
                    }
                }
                SemanticNodeView::Alloc { output } => {
                    let (representation, shape) = self.tensor_type(function, output, assignment)?;
                    environment.insert(
                        output,
                        Value::Tensor(TensorValue::owned(TensorData::uninitialized(
                            representation,
                            shape,
                        )?)),
                    );
                }
                SemanticNodeView::Fill { value, output } => {
                    let (representation, shape) = self.tensor_type(function, output, assignment)?;
                    let RepresentationKind::Dense(dtype) =
                        &registry::representation_info(representation).kind
                    else {
                        unreachable!("checked fill target is not writable dense storage")
                    };
                    let count = shape.iter().product();
                    environment.insert(
                        output,
                        Value::Tensor(TensorValue::owned(TensorData::dense(
                            *dtype,
                            shape,
                            vec![value.value(); count],
                        ))),
                    );
                }
                SemanticNodeView::Copy { input, output } => {
                    let source = self.value(environment, input)?.as_tensor()?.clone();
                    let result = self.copy_tensor(function, output, &source, assignment)?;
                    environment.insert(output, Value::Tensor(result));
                }
                SemanticNodeView::RepresentationConvert {
                    conversion,
                    input,
                    output,
                } => {
                    let source = self.value(environment, input)?.as_tensor()?.clone();
                    let result = self.convert_representation(conversion, &source)?;
                    environment.insert(output, Value::Tensor(result));
                }
                SemanticNodeView::View {
                    base,
                    transform,
                    output,
                } => {
                    let base = self.value(environment, base)?.as_tensor()?.clone();
                    let view = self.apply_view(transform, base, environment, assignment)?;
                    environment.insert(output, Value::Tensor(view));
                }
                SemanticNodeView::ElementRead {
                    place,
                    indices,
                    output,
                } => {
                    let tensor = self.value(environment, place)?.as_tensor()?;
                    let index = self.element_index(tensor, indices, environment)?;
                    let value = self.read_tensor(tensor, index)?;
                    let dtype = scalar_dtype(&function.value(output).ty);
                    environment.insert(output, Value::Scalar(dtype, super::round_to(dtype, value)));
                }
                SemanticNodeView::ElementWrite {
                    place,
                    indices,
                    value,
                    output,
                } => {
                    let tensor = self.value(environment, place)?.as_tensor()?.clone();
                    let index = self.element_index(&tensor, indices, environment)?;
                    let value = self.value(environment, value)?.as_scalar()?;
                    self.write_tensor(&tensor, index, value)?;
                    environment.insert(
                        output,
                        self.updated_base(function, place, &tensor, environment)?,
                    );
                }
                SemanticNodeView::Store {
                    destination,
                    value,
                    output,
                } => {
                    let destination_id = destination;
                    let destination = self
                        .value(environment, destination_id)?
                        .as_tensor()?
                        .clone();
                    let source = self.value(environment, value)?.clone();
                    self.store_tensor(&destination, &source)?;
                    environment.insert(
                        output,
                        self.updated_base(function, destination_id, &destination, environment)?,
                    );
                }
                SemanticNodeView::Atomic {
                    op,
                    place,
                    arguments,
                    output,
                } => {
                    let tensor = self.value(environment, place)?.as_tensor()?.clone();
                    let (value, indices) = arguments
                        .split_last()
                        .unwrap_or_else(|| unreachable!("checked atomic has no value"));
                    let index = self.element_index(&tensor, indices, environment)?;
                    let current = self.read_tensor(&tensor, index)?;
                    let value = self.value(environment, *value)?.as_scalar()?;
                    let dtype = registry::representation_info(tensor.representation).decoded;
                    let next = match op {
                        AtomicOp::Add => scalar::binary(
                            crate::syntax::ast::BinaryOp::Add,
                            (dtype, current),
                            value,
                            Some(dtype),
                        )?,
                        AtomicOp::Max => scalar::math(
                            crate::intrinsics::MathOp::Max,
                            &[(dtype, current), value],
                        )?,
                        AtomicOp::Min => scalar::math(
                            crate::intrinsics::MathOp::Min,
                            &[(dtype, current), value],
                        )?,
                    };
                    self.write_tensor(&tensor, index, next)?;
                    environment.insert(
                        output,
                        self.updated_base(function, place, &tensor, environment)?,
                    );
                }
                SemanticNodeView::If {
                    condition,
                    captures,
                    outputs,
                    then,
                    otherwise,
                } => {
                    let condition = self.value(environment, condition)?.as_scalar()?.1 != 0.0;
                    let child = if condition { then } else { otherwise };
                    let mut child_environment = environment.clone();
                    self.bind_region_parameters(
                        function,
                        child,
                        captures,
                        &mut child_environment,
                        environment,
                    )?;
                    self.execute_region(function, child, &mut child_environment, assignment)?;
                    let results = function.region(child).results();
                    if results.len() != outputs.len() {
                        unreachable!("checked if branch result arity mismatch")
                    }
                    for (output, result) in outputs.iter().zip(results) {
                        environment
                            .insert(*output, self.value(&child_environment, *result)?.clone());
                    }
                }
                SemanticNodeView::Loop {
                    kind,
                    start,
                    end,
                    captures,
                    outputs,
                    body,
                    carries,
                } => {
                    let start = self.value(environment, start)?.as_scalar()?.1 as i64;
                    let end = self.value(environment, end)?.as_scalar()?.1 as i64;
                    let RegionKind::LoopBody { binder_symbol, .. } = function.region(body).kind()
                    else {
                        unreachable!("checked loop body has the wrong region kind")
                    };
                    let mut current = captures
                        .iter()
                        .map(|capture| self.value(environment, *capture).cloned())
                        .collect::<Result<Vec<_>, _>>()?;
                    for coordinate in start..end {
                        let mut child = environment.clone();
                        let parameters = function.region(body).parameters();
                        child.insert(parameters[0], Value::Scalar(DType::I32, coordinate as f64));
                        for (parameter, value) in parameters[1..].iter().zip(&current) {
                            child.insert(*parameter, value.clone());
                        }
                        assignment.bind(*binder_symbol, SymbolValue::Nat(coordinate as u64));
                        self.execute_region(function, body, &mut child, assignment)?;
                        for carry in carries {
                            let parameter = parameters[1..]
                                .iter()
                                .position(|parameter| *parameter == carry.parameter)
                                .unwrap_or_else(|| {
                                    unreachable!("checked carry parameter is not a loop capture")
                                });
                            current[parameter] = self.value(&child, carry.yielded)?.clone();
                        }
                    }
                    if matches!(kind, LoopKind::Parallel) && !carries.is_empty() {
                        unreachable!("checked parallel loop carries reassigned state")
                    }
                    for (output, carry) in outputs.iter().zip(carries) {
                        let parameter = function.region(body).parameters()[1..]
                            .iter()
                            .position(|parameter| *parameter == carry.parameter)
                            .unwrap_or_else(|| {
                                unreachable!("checked carry parameter is not captured")
                            });
                        environment.insert(*output, current[parameter].clone());
                    }
                }
                SemanticNodeView::Check { condition, reason } => {
                    if self.value(environment, condition)?.as_scalar()?.1 == 0.0 {
                        return Err(format!("semantic check failed: {}", check_name(reason)));
                    }
                }
                SemanticNodeView::TuplePack { inputs, output } => {
                    let items = inputs
                        .iter()
                        .map(|input| self.value(environment, *input).cloned())
                        .collect::<Result<Vec<_>, _>>()?;
                    environment.insert(output, Value::Tuple(items));
                }
                SemanticNodeView::TupleGet {
                    tuple,
                    index,
                    output,
                } => {
                    let Value::Tuple(items) = self.value(environment, tuple)? else {
                        unreachable!("checked tuple projection input is not a tuple")
                    };
                    environment.insert(output, items[index as usize].clone());
                }
                SemanticNodeView::Extent {
                    tensor,
                    axis,
                    output,
                } => {
                    let tensor = self.value(environment, tensor)?.as_tensor()?;
                    environment.insert(
                        output,
                        Value::Scalar(DType::I32, tensor.shape[axis as usize] as f64),
                    );
                }
                SemanticNodeView::Intrinsic { .. } => {
                    unreachable!("backend intrinsic occurs in a sealed portable reference body")
                }
            }
        }
        Ok(())
    }

    fn value<'a>(
        &self,
        environment: &'a Environment,
        id: SemanticValueId,
    ) -> Result<&'a Value, String> {
        environment
            .get(&id)
            .ok_or_else(|| "semantic value was used before definition".to_owned())
    }

    fn bind_region_parameters(
        &self,
        function: &SemanticFunction,
        region: crate::ids::RegionId,
        captures: &[SemanticValueId],
        child: &mut Environment,
        parent: &Environment,
    ) -> Result<(), String> {
        let parameters = function.region(region).parameters();
        if parameters.len() != captures.len() {
            unreachable!("checked region capture arity mismatch")
        }
        for (parameter, capture) in parameters.iter().zip(captures) {
            child.insert(*parameter, self.value(parent, *capture)?.clone());
        }
        Ok(())
    }

    fn eval_primitive(
        &self,
        function: &SemanticFunction,
        primitive: &PrimitiveId,
        inputs: &[SemanticValueId],
        output: SemanticValueId,
        environment: &Environment,
        assignment: &Assignment,
    ) -> Result<Value, String> {
        match primitive {
            PrimitiveId::RangeMake => Ok(Value::Range(
                self.value(environment, inputs[0])?.as_scalar()?.1 as i64,
                self.value(environment, inputs[1])?.as_scalar()?.1 as i64,
            )),
            PrimitiveId::RangeStart | PrimitiveId::RangeEnd => {
                let Value::Range(start, end) = self.value(environment, inputs[0])? else {
                    unreachable!("checked range projection input is not a range")
                };
                Ok(Value::Scalar(
                    DType::I32,
                    if matches!(primitive, PrimitiveId::RangeStart) {
                        *start
                    } else {
                        *end
                    } as f64,
                ))
            }
            PrimitiveId::Symbolic(expression) => {
                let mut values = assignment.clone();
                for symbol in self.entry.arena().free_symbols((*expression).into()) {
                    if let SymbolKind::RuntimeValue(value) = self.entry.arena().symbol_kind(symbol)
                    {
                        let scalar = self.value(environment, value)?.as_scalar()?;
                        values.bind(symbol, scalar_symbol(scalar.0, scalar.1));
                    }
                }
                let value = self
                    .entry
                    .arena()
                    .eval_int(*expression, &values)
                    .map_err(|error| format!("symbolic reference evaluation failed: {error:?}"))?;
                Ok(Value::Scalar(DType::I32, value as f64))
            }
            PrimitiveId::Constant(constant) => {
                let dtype = scalar_dtype(&function.value(output).ty);
                let value = match constant {
                    Constant::Int(value) => *value as f64,
                    Constant::Float(value) => *value,
                    Constant::Bool(value) => f64::from(u8::from(*value)),
                };
                Ok(Value::Scalar(dtype, super::round_to(dtype, value)))
            }
            PrimitiveId::Unary(operation) => Ok(Value::scalar(scalar::unary(
                *operation,
                self.value(environment, inputs[0])?.as_scalar()?,
            )?)),
            PrimitiveId::Binary(operation) => Ok(Value::scalar(scalar::binary(
                *operation,
                self.value(environment, inputs[0])?.as_scalar()?,
                self.value(environment, inputs[1])?.as_scalar()?,
                Some(scalar_dtype(&function.value(output).ty)),
            )?)),
            PrimitiveId::Cast(dtype) => Ok(Value::scalar(scalar::cast(
                *dtype,
                self.value(environment, inputs[0])?.as_scalar()?,
            ))),
            PrimitiveId::Math(operation) => {
                let arguments = inputs
                    .iter()
                    .map(|input| self.value(environment, *input)?.as_scalar())
                    .collect::<Result<Vec<_>, _>>()?;
                Ok(Value::scalar(scalar::math(*operation, &arguments)?))
            }
            PrimitiveId::Select => {
                let condition = self.value(environment, inputs[0])?.as_scalar()?.1 != 0.0;
                Ok(self
                    .value(environment, inputs[if condition { 1 } else { 2 }])?
                    .clone())
            }
            PrimitiveId::TuplePack | PrimitiveId::TupleGet(_) => {
                unreachable!("tuple primitive survived semantic canonicalization")
            }
            PrimitiveId::TensorAlloc
            | PrimitiveId::Fill(_)
            | PrimitiveId::Materialize
            | PrimitiveId::Clone
            | PrimitiveId::Load
            | PrimitiveId::RepresentationConvert(_)
            | PrimitiveId::Transpose
            | PrimitiveId::Reshape
            | PrimitiveId::SliceView { .. }
            | PrimitiveId::ElementRead { .. }
            | PrimitiveId::Extent { .. }
            | PrimitiveId::Atomic { .. }
            | PrimitiveId::Reduce { .. }
            | PrimitiveId::Decode => {
                unreachable!("structural primitive survived semantic canonicalization")
            }
        }
    }

    fn eval_elementwise(
        &self,
        function: &SemanticFunction,
        primitive: &PrimitiveId,
        inputs: &[SemanticValueId],
        output: SemanticValueId,
        environment: &Environment,
        assignment: &Assignment,
    ) -> Result<Value, String> {
        let (representation, shape) = self.tensor_type(function, output, assignment)?;
        let RepresentationKind::Dense(dtype) = &registry::representation_info(representation).kind
        else {
            unreachable!("computed elementwise result is not dense")
        };
        let count = shape.iter().product();
        let mut data = Vec::with_capacity(count);
        for index in 0..count {
            let mut scalar_inputs = Vec::with_capacity(inputs.len());
            for input in inputs {
                let value = self.value(environment, *input)?;
                let scalar = match value {
                    Value::Scalar(..) => value.clone(),
                    Value::Tensor(tensor) => Value::Scalar(
                        registry::representation_info(tensor.representation).decoded,
                        self.read_tensor(tensor, index)?,
                    ),
                    _ => unreachable!("checked elementwise operand is not numerical"),
                };
                scalar_inputs.push(scalar.as_scalar()?);
            }
            data.push(apply_scalar_primitive(primitive, &scalar_inputs, *dtype)?.1);
        }
        Ok(Value::Tensor(TensorValue::owned(TensorData::dense(
            *dtype, shape, data,
        ))))
    }

    fn eval_reduce(
        &self,
        function: &SemanticFunction,
        operation: ReduceOp,
        axis: u32,
        input: SemanticValueId,
        output: SemanticValueId,
        environment: &Environment,
        assignment: &Assignment,
    ) -> Result<Value, String> {
        let input = self.value(environment, input)?.as_tensor()?;
        let axis = axis as usize;
        let extent = input.shape[axis];
        let output_shape = match &function.value(output).ty {
            SemanticType::Tensor(_) => self.tensor_type(function, output, assignment)?.1,
            _ => Vec::new(),
        };
        let input_dtype = registry::representation_info(input.representation).decoded;
        let output_dtype = match &function.value(output).ty {
            SemanticType::Tensor(tensor) => {
                registry::representation_info(tensor.representation).decoded
            }
            ty => scalar_dtype(ty),
        };
        let output_count: usize = output_shape.iter().product();
        let input_strides = row_major(&input.shape);
        let output_strides = row_major(&output_shape);
        let mut results = Vec::with_capacity(output_count.max(1));
        for output_flat in 0..output_count.max(1) {
            let mut remainder = output_flat;
            let mut output_coordinate = vec![0; output_shape.len()];
            for (coordinate, stride) in output_coordinate.iter_mut().zip(&output_strides) {
                *coordinate = remainder / stride;
                remainder %= stride;
            }
            let mut best_index = 0usize;
            let mut accumulator = match operation {
                ReduceOp::Sum => 0.0,
                ReduceOp::Max | ReduceOp::Argmax => f64::NEG_INFINITY,
                ReduceOp::Min => f64::INFINITY,
            };
            for coordinate in 0..extent {
                let mut input_coordinate = output_coordinate.clone();
                input_coordinate.insert(axis, coordinate);
                let logical = input_coordinate
                    .iter()
                    .zip(&input_strides)
                    .map(|(coordinate, stride)| coordinate * stride)
                    .sum();
                let value = self.read_tensor(input, logical)?;
                match operation {
                    ReduceOp::Sum => {
                        let dtype = accumulator_dtype(operation, input_dtype);
                        accumulator = scalar::binary(
                            crate::syntax::ast::BinaryOp::Add,
                            (dtype, accumulator),
                            (input_dtype, value),
                            Some(dtype),
                        )?
                        .1;
                    }
                    ReduceOp::Max => accumulator = accumulator.max(value),
                    ReduceOp::Min => accumulator = accumulator.min(value),
                    ReduceOp::Argmax if value > accumulator => {
                        accumulator = value;
                        best_index = coordinate;
                    }
                    ReduceOp::Argmax => {}
                }
            }
            results.push(if matches!(operation, ReduceOp::Argmax) {
                best_index as f64
            } else {
                super::round_to(output_dtype, accumulator)
            });
        }
        if matches!(function.value(output).ty, SemanticType::Tensor(_)) {
            Ok(Value::Tensor(TensorValue::owned(TensorData::dense(
                output_dtype,
                output_shape,
                results,
            ))))
        } else {
            Ok(Value::Scalar(output_dtype, results[0]))
        }
    }

    fn tensor_type(
        &self,
        function: &SemanticFunction,
        value: SemanticValueId,
        assignment: &Assignment,
    ) -> Result<(crate::ids::RepresentationId, Vec<usize>), String> {
        let SemanticType::Tensor(tensor) = &function.value(value).ty else {
            unreachable!("checked tensor operation has non-tensor output")
        };
        let shape = tensor
            .axes
            .iter()
            .map(|axis| {
                self.entry
                    .arena()
                    .eval_nat(*axis, assignment)
                    .map_err(|error| format!("tensor extent evaluation failed: {error:?}"))
                    .and_then(|extent| {
                        usize::try_from(extent)
                            .map_err(|_| "tensor extent exceeds usize".to_owned())
                    })
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok((tensor.representation, shape))
    }

    fn read_tensor(&self, tensor: &TensorValue, logical: usize) -> Result<f64, String> {
        let flat = *tensor
            .positions
            .get(logical)
            .ok_or("tensor index outside logical shape")?;
        match &tensor.backing {
            Backing::Argument(index) => self.tensors[*index].read(flat),
            Backing::Owned(data) => data.borrow().read(flat),
        }
    }

    fn write_tensor(
        &mut self,
        tensor: &TensorValue,
        logical: usize,
        value: Scalar,
    ) -> Result<(), String> {
        let flat = *tensor
            .positions
            .get(logical)
            .ok_or("tensor index outside logical shape")?;
        match &tensor.backing {
            Backing::Argument(index) => self.tensors[*index].write(flat, value),
            Backing::Owned(data) => data.borrow_mut().write(flat, value),
        }
    }

    fn store_tensor(&mut self, destination: &TensorValue, source: &Value) -> Result<(), String> {
        let values = match source {
            Value::Scalar(dtype, value) => vec![(*dtype, *value); destination.element_count()],
            Value::Tensor(source) => {
                if source.shape != destination.shape {
                    return Err("tensor store shape mismatch".to_owned());
                }
                (0..source.element_count())
                    .map(|index| {
                        Ok((
                            registry::representation_info(source.representation).decoded,
                            self.read_tensor(source, index)?,
                        ))
                    })
                    .collect::<Result<Vec<_>, String>>()?
            }
            _ => return Err("tensor store source is not numerical".to_owned()),
        };
        for (index, value) in values.into_iter().enumerate() {
            self.write_tensor(destination, index, value)?;
        }
        Ok(())
    }

    fn element_index(
        &self,
        tensor: &TensorValue,
        indices: &[SemanticValueId],
        environment: &Environment,
    ) -> Result<usize, String> {
        if indices.len() != tensor.shape.len() {
            unreachable!("checked point access index arity differs from tensor rank")
        }
        let strides = row_major(&tensor.shape);
        let mut flat = 0usize;
        for ((index, extent), stride) in indices.iter().zip(&tensor.shape).zip(strides) {
            let index = self.value(environment, *index)?.as_scalar()?.1 as i64;
            if index < 0 || index as usize >= *extent {
                return Err("tensor index outside logical shape".to_owned());
            }
            flat += index as usize * stride;
        }
        Ok(flat)
    }

    fn copy_tensor(
        &self,
        function: &SemanticFunction,
        output: SemanticValueId,
        source: &TensorValue,
        assignment: &Assignment,
    ) -> Result<TensorValue, String> {
        let (representation, shape) = self.tensor_type(function, output, assignment)?;
        if representation == source.representation
            && source
                .positions
                .iter()
                .copied()
                .eq(0..source.element_count())
        {
            let data = match &source.backing {
                Backing::Argument(index) => self.tensors[*index].clone(),
                Backing::Owned(data) => data.borrow().clone(),
            };
            return Ok(TensorValue::owned(data));
        }
        let RepresentationKind::Dense(dtype) = &registry::representation_info(representation).kind
        else {
            return Err("non-dense view copy requires a complete identity packet view".to_owned());
        };
        let values = (0..source.element_count())
            .map(|index| self.read_tensor(source, index))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(TensorValue::owned(TensorData::dense(*dtype, shape, values)))
    }

    fn apply_view(
        &self,
        transform: &ViewTransform,
        mut base: TensorValue,
        environment: &Environment,
        assignment: &Assignment,
    ) -> Result<TensorValue, String> {
        match transform {
            ViewTransform::Identity => Ok(base),
            ViewTransform::Plane { .. } => {
                Err("raw packed planes have no portable reference value".to_owned())
            }
            ViewTransform::Reshape { axes } => {
                let shape = axes
                    .iter()
                    .map(|axis| {
                        self.entry
                            .arena()
                            .eval_nat(*axis, assignment)
                            .map(|value| value as usize)
                            .map_err(|error| format!("reshape extent failed: {error:?}"))
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                if shape.iter().product::<usize>() != base.element_count() {
                    unreachable!("checked reshape changes element count")
                }
                base.shape = shape;
                Ok(base)
            }
            ViewTransform::Transpose { permutation } => {
                let old_shape = base.shape.clone();
                let old_strides = row_major(&old_shape);
                let new_shape = permutation
                    .iter()
                    .map(|axis| old_shape[*axis as usize])
                    .collect::<Vec<_>>();
                let new_strides = row_major(&new_shape);
                let mut positions = Vec::with_capacity(base.element_count());
                for flat in 0..base.element_count() {
                    let mut remainder = flat;
                    let mut new_coordinate = vec![0; new_shape.len()];
                    for (coordinate, stride) in new_coordinate.iter_mut().zip(&new_strides) {
                        *coordinate = remainder / stride;
                        remainder %= stride;
                    }
                    let mut old_coordinate = vec![0; old_shape.len()];
                    for (new_axis, old_axis) in permutation.iter().enumerate() {
                        old_coordinate[*old_axis as usize] = new_coordinate[new_axis];
                    }
                    let old_flat = old_coordinate
                        .iter()
                        .zip(&old_strides)
                        .map(|(a, b)| a * b)
                        .sum::<usize>();
                    positions.push(base.positions[old_flat]);
                }
                base.shape = new_shape;
                base.positions = positions;
                Ok(base)
            }
            ViewTransform::Slice { axes } => {
                let old_shape = base.shape.clone();
                let old_strides = row_major(&old_shape);
                let mut choices = Vec::new();
                let mut output_shape = Vec::new();
                for (axis, transform) in axes.iter().enumerate() {
                    let extent = old_shape[axis];
                    match transform {
                        crate::entry::SliceAxis::Full => {
                            choices.push((0..extent).collect::<Vec<_>>());
                            output_shape.push(extent);
                        }
                        crate::entry::SliceAxis::Point(value) => {
                            let value = self.scalar_ref(value, environment, assignment)?;
                            choices.push(vec![value]);
                        }
                        crate::entry::SliceAxis::Range { start, end } => {
                            let start = start
                                .as_ref()
                                .map(|value| self.scalar_ref(value, environment, assignment))
                                .transpose()?
                                .unwrap_or(0);
                            let end = end
                                .as_ref()
                                .map(|value| self.scalar_ref(value, environment, assignment))
                                .transpose()?
                                .unwrap_or(extent);
                            choices.push((start..end).collect());
                            output_shape.push(end - start);
                        }
                    }
                }
                for extent in &old_shape[axes.len()..] {
                    choices.push((0..*extent).collect());
                    output_shape.push(*extent);
                }
                let mut positions = Vec::with_capacity(output_shape.iter().product());
                enumerate_coordinates(&choices, 0, &mut Vec::new(), &mut |coordinate| {
                    let old_flat = coordinate
                        .iter()
                        .zip(&old_strides)
                        .map(|(a, b)| a * b)
                        .sum::<usize>();
                    positions.push(base.positions[old_flat]);
                });
                base.shape = output_shape;
                base.positions = positions;
                Ok(base)
            }
        }
    }

    fn scalar_ref(
        &self,
        value: &ScalarRef,
        environment: &Environment,
        assignment: &Assignment,
    ) -> Result<usize, String> {
        match value {
            ScalarRef::Static(value) => self
                .entry
                .arena()
                .eval_nat(*value, assignment)
                .map(|value| value as usize)
                .map_err(|error| format!("slice bound failed: {error:?}")),
            ScalarRef::Value(value) => Ok(self.value(environment, *value)?.as_scalar()?.1 as usize),
        }
    }

    fn updated_base(
        &self,
        function: &SemanticFunction,
        place: SemanticValueId,
        written: &TensorValue,
        environment: &Environment,
    ) -> Result<Value, String> {
        let mut current = Some(place);
        while let Some(value) = current {
            if let Ok(candidate) = self.value(environment, value).and_then(Value::as_tensor) {
                if same_backing(candidate, written) {
                    return Ok(Value::Tensor(candidate.clone()));
                }
            }
            current = match &function.value(value).ty {
                SemanticType::Tensor(tensor) => match &tensor.storage {
                    TensorStorage::View { base, .. } => Some(*base),
                    _ => None,
                },
                _ => None,
            };
        }
        unreachable!("checked writable view has no bound storage base")
    }

    fn convert_representation(
        &self,
        conversion: RepresentationConversionId,
        source: &TensorValue,
    ) -> Result<TensorValue, String> {
        let conversion = registry::representation_conversion_info(conversion);
        if source.representation != conversion.source {
            unreachable!("checked representation conversion source mismatch")
        }
        if !source
            .positions
            .iter()
            .copied()
            .eq(0..source.element_count())
        {
            unreachable!("representation conversion source is not a complete tensor")
        }
        let source_data = match &source.backing {
            Backing::Argument(index) => &self.tensors[*index],
            Backing::Owned(data) => {
                return convert_owned_encoded(conversion.id, source.shape.clone(), &data.borrow())
            }
        };
        convert_owned_encoded(conversion.id, source.shape.clone(), source_data)
    }
}

fn convert_owned_encoded(
    conversion: RepresentationConversionId,
    shape: Vec<usize>,
    source: &TensorData,
) -> Result<TensorValue, String> {
    let conversion = registry::representation_conversion_info(conversion);
    let (representation, _, source_bytes) = source
        .encoded_parts()
        .ok_or("representation conversion source is not encoded")?;
    if representation != conversion.source {
        unreachable!("checked representation conversion source mismatch")
    }
    let source_layout = match &registry::representation_info(conversion.source).kind {
        RepresentationKind::External(layout) => layout,
        _ => unreachable!("registered conversion source is not external"),
    };
    let destination_layout = match &registry::representation_info(conversion.destination).kind {
        RepresentationKind::Packed(layout) => layout,
        _ => unreachable!("registered conversion destination is not packed"),
    };
    let packet_count = source_bytes.len() / source_layout.packet_size as usize;
    let mut destination = vec![0u8; packet_count * destination_layout.packet_size as usize];
    for packet in 0..packet_count {
        let source_packet = &source_bytes[packet * source_layout.packet_size as usize
            ..(packet + 1) * source_layout.packet_size as usize];
        let destination_packet = &mut destination[packet * destination_layout.packet_size as usize
            ..(packet + 1) * destination_layout.packet_size as usize];
        for (plane, recipe) in destination_layout
            .planes
            .iter()
            .zip(&conversion.recipe.planes)
        {
            let plane_bytes = &mut destination_packet
                [plane.offset as usize..plane.offset as usize + plane.bytes_per_group as usize];
            match recipe {
                PlaneRepackRecipe::BitRoutes(routes) => {
                    for (destination_bit, source_bit) in routes.iter().enumerate() {
                        let bit = read_bits(source_packet, *source_bit as usize, 1);
                        write_bits(plane_bytes, destination_bit, 1, bit);
                    }
                }
                PlaneRepackRecipe::DenseValues(expressions) => {
                    for (index, expression) in expressions.iter().enumerate() {
                        let value = eval_repack(expression, source_packet);
                        let width = plane.storage_dtype.bytes() as usize;
                        let bytes = &mut plane_bytes[index * width..(index + 1) * width];
                        write_dense(plane.storage_dtype, value, bytes);
                    }
                }
            }
        }
    }
    Ok(TensorValue::owned(TensorData::encoded(
        conversion.destination,
        shape,
        destination,
    )?))
}

fn eval_repack(expression: &RepackExpr, source: &[u8]) -> f64 {
    match expression {
        RepackExpr::SourceBits { bit, width } => {
            f64::from(read_bits(source, *bit as usize, u32::from(*width)))
        }
        RepackExpr::ShiftLeft { value, bits } => {
            ((eval_repack(value, source) as u32) << bits) as f64
        }
        RepackExpr::BitOr(left, right) => {
            ((eval_repack(left, source) as u32) | (eval_repack(right, source) as u32)) as f64
        }
        RepackExpr::OffsetI32 { value, offset } => {
            (eval_repack(value, source) as i32 + offset) as f64
        }
        RepackExpr::F16ToF32(value) => {
            super::tensor::f16_to_f32(eval_repack(value, source) as u16) as f64
        }
        RepackExpr::I32ToF32(value) => eval_repack(value, source) as i32 as f32 as f64,
        RepackExpr::MultiplyF32(left, right) => {
            (eval_repack(left, source) as f32 * eval_repack(right, source) as f32) as f64
        }
    }
}

fn write_dense(dtype: DType, value: f64, bytes: &mut [u8]) {
    match dtype {
        DType::F32 => bytes.copy_from_slice(&(value as f32).to_le_bytes()),
        DType::F16 => bytes.copy_from_slice(&super::tensor::f16_bits(value as f32).to_le_bytes()),
        DType::BF16 => bytes.copy_from_slice(
            &((super::tensor::bf16_round(value as f32).to_bits() >> 16) as u16).to_le_bytes(),
        ),
        DType::I32 => bytes.copy_from_slice(&(value as i32).to_le_bytes()),
        DType::U32 => bytes.copy_from_slice(&(value as u32).to_le_bytes()),
        DType::Bool => bytes[0] = u8::from(value != 0.0),
    }
}

fn same_backing(left: &TensorValue, right: &TensorValue) -> bool {
    match (&left.backing, &right.backing) {
        (Backing::Argument(left), Backing::Argument(right)) => left == right,
        (Backing::Owned(left), Backing::Owned(right)) => std::rc::Rc::ptr_eq(left, right),
        _ => false,
    }
}

fn scalar_dtype(ty: &SemanticType) -> DType {
    match ty {
        SemanticType::Scalar(dtype) => *dtype,
        SemanticType::Index { .. } => DType::I32,
        _ => unreachable!("checked scalar operation has non-scalar type"),
    }
}

fn scalar_symbol(dtype: DType, value: f64) -> SymbolValue {
    match dtype {
        DType::F32 => SymbolValue::F32(value as f32),
        DType::F16 => SymbolValue::F16(super::tensor::f16_bits(value as f32)),
        DType::BF16 => {
            SymbolValue::BF16((super::tensor::bf16_round(value as f32).to_bits() >> 16) as u16)
        }
        DType::I32 => SymbolValue::I32(value as i32),
        DType::U32 => SymbolValue::U32(value as u32),
        DType::Bool => SymbolValue::Bool(value != 0.0),
    }
}

fn apply_scalar_primitive(
    primitive: &PrimitiveId,
    arguments: &[Scalar],
    output: DType,
) -> Result<Scalar, String> {
    match primitive {
        PrimitiveId::Constant(value) => Ok((
            output,
            super::round_to(
                output,
                match value {
                    Constant::Int(value) => *value as f64,
                    Constant::Float(value) => *value,
                    Constant::Bool(value) => f64::from(u8::from(*value)),
                },
            ),
        )),
        PrimitiveId::Unary(operation) => scalar::unary(*operation, arguments[0]),
        PrimitiveId::Binary(operation) => {
            scalar::binary(*operation, arguments[0], arguments[1], Some(output))
        }
        PrimitiveId::Cast(dtype) => Ok(scalar::cast(*dtype, arguments[0])),
        PrimitiveId::Math(operation) => scalar::math(*operation, arguments),
        PrimitiveId::Select => Ok(if arguments[0].1 != 0.0 {
            arguments[1]
        } else {
            arguments[2]
        }),
        PrimitiveId::Decode => Ok((output, super::round_to(output, arguments[0].1))),
        PrimitiveId::Symbolic(_)
        | PrimitiveId::TuplePack
        | PrimitiveId::TupleGet(_)
        | PrimitiveId::RangeMake
        | PrimitiveId::RangeStart
        | PrimitiveId::RangeEnd
        | PrimitiveId::TensorAlloc
        | PrimitiveId::Fill(_)
        | PrimitiveId::Materialize
        | PrimitiveId::Clone
        | PrimitiveId::Load
        | PrimitiveId::RepresentationConvert(_)
        | PrimitiveId::Transpose
        | PrimitiveId::Reshape
        | PrimitiveId::SliceView { .. }
        | PrimitiveId::ElementRead { .. }
        | PrimitiveId::Extent { .. }
        | PrimitiveId::Atomic { .. }
        | PrimitiveId::Reduce { .. } => {
            unreachable!("non-scalar primitive occurs in a checked elementwise node")
        }
    }
}

fn check_name(reason: &CheckReason) -> &str {
    match reason {
        CheckReason::IndexBound => "index bound",
        CheckReason::RangeOrder => "range order",
        CheckReason::DivideByZero => "divide by zero",
        CheckReason::SignedDivisionOverflow => "signed division overflow",
        CheckReason::Custom(reason) => reason,
    }
}

fn enumerate_coordinates(
    choices: &[Vec<usize>],
    axis: usize,
    coordinate: &mut Vec<usize>,
    visit: &mut dyn FnMut(&[usize]),
) {
    if axis == choices.len() {
        visit(coordinate);
        return;
    }
    for value in &choices[axis] {
        coordinate.push(*value);
        enumerate_coordinates(choices, axis + 1, coordinate, visit);
        coordinate.pop();
    }
}
