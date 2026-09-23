use super::scalar;
use super::tensor::{read_bits, write_bits};
use super::value::{row_major, Backing, TensorValue, Value};
use super::{Arg, EvalError, Interpreter, OracleError, TensorData};
use crate::entry::{
    Candidate, LoopKind, NumericalRole, ParameterKind, RegionKind, ScalarRef, SemanticFunction,
    SemanticNodeView, SemanticType, TensorStorage, ViewTransform,
};
use crate::expr::{
    compiled::InvocationValues, Assignment, PartialAssignment, SymbolKind, SymbolValue,
};
use crate::failure::{SourceFailure, SourceFailureCause};
use crate::ids::{FamilyId, RepresentationConversionId, SemanticValueId};
use crate::intrinsics::{accumulator_dtype, AtomicOp, PrimitiveId, ReduceOp};
use crate::reference_math::ReferenceScalar;
use crate::registry::{self, PlaneRepackRecipe, RepackExpr, RepresentationKind};
use crate::types::DType;
use num_bigint::BigInt;
use num_traits::{Euclid, ToPrimitive, Zero};
use std::collections::BTreeMap;

type Environment = BTreeMap<SemanticValueId, Value>;

impl Interpreter<'_> {
    fn bind_arguments(
        &mut self,
        arguments: &[Arg],
    ) -> Result<(Vec<Value>, Assignment), OracleError> {
        if arguments.len() != self.entry.schema().parameters().len() {
            return Err(OracleError::InvalidInvocation(format!(
                "entry expects {} flattened arguments, got {}",
                self.entry.schema().parameters().len(),
                arguments.len()
            )));
        }
        for alias in self.entry.schema().aliases() {
            if let crate::entry::AliasRule::Disjoint(left, right) = alias {
                let left = &arguments[self.entry.schema().parameter_ordinal(*left)];
                let right = &arguments[self.entry.schema().parameter_ordinal(*right)];
                if let (Arg::Tensor(a), Arg::Tensor(b)) = (left, right) {
                    if a == b {
                        let tensor = self.tensors.get(*a).ok_or_else(|| {
                            OracleError::InvalidInvocation(
                                "tensor argument index is outside the oracle table".into(),
                            )
                        })?;
                        if tensor.storage_bytes()? != 0 {
                            return Err(OracleError::InvalidInvocation(
                                "disjoint tensor parameters share storage".into(),
                            ));
                        }
                    }
                }
            }
        }
        let mut assignment = Assignment::new();
        let mut values = Vec::with_capacity(arguments.len());
        for (parameter, argument) in self.entry.schema().parameters().iter().zip(arguments) {
            let value = match (&parameter.kind, argument) {
                (ParameterKind::Tensor { representation, .. }, Arg::Tensor(index)) => {
                    let tensor = self.tensors.get(*index).ok_or_else(|| {
                        OracleError::InvalidInvocation(
                            "tensor argument index is outside the oracle table".into(),
                        )
                    })?;
                    if tensor.representation() != *representation {
                        return Err(OracleError::InvalidInvocation(format!(
                            "tensor argument `{}` has representation `{}`, expected `{}`",
                            parameter.name,
                            registry::representation_info(tensor.representation()).name,
                            registry::representation_info(*representation).name
                        )));
                    }
                    self.charge_tensor(tensor.shape())?;
                    Value::Tensor(TensorValue::argument(
                        *index,
                        *representation,
                        tensor.shape(),
                        self.reserve_tensor_value(*representation, tensor.shape(), false)?,
                    ))
                }
                (ParameterKind::Scalar { dtype, symbol }, Arg::Scalar(value)) => {
                    if *dtype != value.dtype() {
                        return Err(OracleError::InvalidInvocation(format!(
                            "scalar argument `{}` has the wrong dtype",
                            parameter.name
                        )));
                    }
                    assignment.bind(*symbol, scalar_symbol(*value));
                    Value::Scalar(*value)
                }
                (ParameterKind::Index { symbol, .. }, Arg::Index(value)) => {
                    assignment.bind(*symbol, SymbolValue::Nat(value.clone()));
                    Value::Index(value.clone())
                }
                (ParameterKind::Range { start, end, .. }, Arg::Range(first, last)) => {
                    assignment.bind(*start, SymbolValue::Nat(first.clone()));
                    assignment.bind(*end, SymbolValue::Nat(last.clone()));
                    Value::Range(first.clone(), last.clone())
                }
                _ => {
                    return Err(OracleError::InvalidInvocation(format!(
                        "argument kind mismatch for `{}`",
                        parameter.name
                    )))
                }
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
                .ok_or_else(|| {
                    OracleError::InvalidInvocation(
                        "tensor argument index is outside the oracle table".into(),
                    )
                })?
                .shape();
            if shape.len() != axes.len() {
                return Err(OracleError::InvalidInvocation(format!(
                    "tensor argument `{}` has the wrong rank",
                    parameter.name
                )));
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
                OracleError::InvalidInvocation(format!(
                    "tensor-axis observation {} does not admit the entry's exact dimension solution",
                    failure.observation()
                ))
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
            return Err(OracleError::InvalidInvocation(
                "invocation is outside the entry domain".into(),
            ));
        }

        Ok((values, assignment))
    }

    pub(super) fn run_reference(&mut self, arguments: &[Arg]) -> Result<Vec<Value>, EvalError> {
        let (values, mut assignment) = self.bind_arguments(arguments)?;
        let root = self.entry.program().root();
        let results = self.call_reference(root, values, &mut assignment)?;
        if results.len() != self.entry.schema().results().len() {
            unreachable!("checked root reference result arity differs from its call schema")
        }
        Ok(results)
    }

    fn reference<'a>(
        entry: crate::entry::LogicalEntryView<'a>,
        family: FamilyId,
    ) -> (&'a Candidate, &'a SemanticFunction) {
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
    ) -> Result<Vec<Value>, EvalError> {
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
                environment.get(result).cloned().ok_or_else(|| {
                    EvalError::from("reference body did not define a declared result")
                })
            })
            .collect()
    }

    fn execute_region(
        &mut self,
        function: &SemanticFunction,
        region: crate::ids::RegionId,
        environment: &mut Environment,
        assignment: &mut Assignment,
    ) -> Result<(), EvalError> {
        for (node_id, node) in function.nodes(region) {
            self.charge_work(1)?;
            for event in node.events() {
                if let crate::entry::NumericalOutcome::AllowedAssociation(outcome) =
                    event.numerical_outcome()
                {
                    self.record_association(node_id, *outcome)?;
                }
            }
            if let SemanticNodeView::Reduce {
                op,
                unordered: true,
                input,
                ..
            } = node.view()
            {
                let dtype = match &function.value(input).ty {
                    SemanticType::Tensor(t) => {
                        registry::representation_info(t.representation).decoded
                    }
                    SemanticType::Scalar(dtype) => *dtype,
                    _ => unreachable!("checked reduction requires numeric input"),
                };
                let accumulator = accumulator_dtype(op, dtype);
                if accumulator.is_float() {
                    self.record_association(
                        node_id,
                        crate::entry::AssociationOutcome::Reassociated { accumulator },
                    )?;
                }
            }
            let result: Result<(), EvalError> = (|| {
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
                    SemanticNodeView::Alloc { extents, output } => {
                        let SemanticType::Tensor(tensor) = &function.value(output).ty else {
                            unreachable!("checked allocation has a non-tensor result")
                        };
                        let representation = tensor.representation;
                        let shape = extents
                            .iter()
                            .map(|extent| self.value(environment, *extent)?.as_nat_usize().map_err(EvalError::from))
                            .collect::<Result<Vec<_>, _>>()?;
                        self.charge_tensor(&shape)?;
                        let memory = self.reserve_tensor_value(representation, &shape, true)?;
                        environment.insert(
                            output,
                            Value::Tensor(TensorValue::owned(
                                TensorData::uninitialized(representation, shape)?,
                                memory,
                            )),
                        );
                    }
                    SemanticNodeView::Fill { value, like, output } => {
                        let SemanticType::Tensor(tensor) = &function.value(output).ty else {
                            unreachable!("checked fill has a non-tensor result")
                        };
                        let representation = tensor.representation;
                        let shape = self.value(environment, like)?.as_tensor()?.shape().to_vec();
                        self.charge_tensor(&shape)?;
                        let RepresentationKind::Dense(dtype) =
                            &registry::representation_info(representation).kind
                        else {
                            unreachable!("checked fill target is not writable dense storage")
                        };
                        let count = shape.iter().product();
                        let memory = self.reserve_tensor_value(representation, &shape, true)?;
                        let mut data = TensorData::uninitialized(representation, shape)?;
                        for index in 0..count {
                            data.write(
                                index,
                                crate::reference_math::integer_literal(
                                    *dtype,
                                    value.value() as i128,
                                ),
                            )?;
                        }
                        environment.insert(output, Value::Tensor(TensorValue::owned(data, memory)));
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
                        extents,
                        transform,
                        output,
                    } => {
                        let base = self.value(environment, base)?.as_tensor()?.clone();
                        let view = self.apply_view(transform, extents, base, environment, assignment)?;
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
                        environment.insert(output, Value::Scalar(scalar::cast(dtype, value)));
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
                        environment.insert(output, Value::Tensor(tensor));
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
                        environment.insert(output, {
                            let SemanticType::Tensor(tensor) = &function.value(destination_id).ty
                            else {
                                unreachable!("checked store destination is a tensor")
                            };
                            let TensorStorage::View { base, .. } = tensor.storage else {
                                unreachable!("checked store destination is an explicit view")
                            };
                            let base = self.value(environment, base)?.as_tensor()?.clone();
                            assert!(
                                same_backing(&base, &destination),
                                "store view lost its actual backing"
                            );
                            Value::Tensor(base)
                        });
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
                                current,
                                value,
                                Some(dtype),
                            )?,
                            AtomicOp::Max => {
                                scalar::math(crate::intrinsics::MathOp::Max, &[current, value])?
                            }
                            AtomicOp::Min => {
                                scalar::math(crate::intrinsics::MathOp::Min, &[current, value])?
                            }
                        };
                        self.write_tensor(&tensor, index, next)?;
                        environment.insert(output, Value::Tensor(tensor));
                    }
                    SemanticNodeView::If {
                        condition,
                        captures,
                        outputs,
                        then,
                        otherwise,
                    } => {
                        let condition =
                            self.value(environment, condition)?.as_scalar()?.bits() != 0;
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
                        let start = self.value(environment, start)?.as_nat()?;
                        let end = self.value(environment, end)?.as_nat()?;
                        let RegionKind::LoopBody { binder_symbol, .. } =
                            function.region(body).kind()
                        else {
                            unreachable!("checked loop body has the wrong region kind")
                        };
                        let mut current = captures
                            .iter()
                            .map(|capture| self.value(environment, *capture).cloned())
                            .collect::<Result<Vec<_>, _>>()?;
                        if matches!(kind, LoopKind::Parallel) && start < end {
                            self.record_parallel(node_id)?;
                        }
                        let mut coordinate = start;
                        while coordinate < end {
                            self.charge_work(1)?;
                            let mut child = environment.clone();
                            let parameters = function.region(body).parameters();
                            child.insert(parameters[0], Value::Index(coordinate.clone()));
                            for (parameter, value) in parameters[1..].iter().zip(&current) {
                                child.insert(*parameter, value.clone());
                            }
                            assignment.bind(*binder_symbol, SymbolValue::Nat(coordinate.clone()));
                            self.execute_region(function, body, &mut child, assignment)?;
                            for carry in carries {
                                let parameter = parameters[1..]
                                    .iter()
                                    .position(|parameter| *parameter == carry.parameter)
                                    .unwrap_or_else(|| {
                                        unreachable!(
                                            "checked carry parameter is not a loop capture"
                                        )
                                    });
                                current[parameter] = self.value(&child, carry.yielded)?.clone();
                            }
                            coordinate += 1u8;
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
                        if self.value(environment, condition)?.as_scalar()?.bits() == 0 {
                            return Err(EvalError::Source(SourceFailure::at(
                                function,
                                node_id,
                                SourceFailureCause::Check(reason.clone()),
                            )));
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
                        let extent = tensor.shape[axis as usize];
                        let value = match function.value(output).ty {
                            SemanticType::Integer => Value::Integer(extent.into()),
                            // Authored extent keeps its existing source value
                            // path; only the hidden call-dimension capture is
                            // an exact mathematical Integer result.
                            _ => Value::Index(extent.into()),
                        };
                        environment.insert(output, value);
                    }
                    SemanticNodeView::Intrinsic { .. } => {
                        unreachable!("backend intrinsic occurs in a sealed portable reference body")
                    }
                }
                Ok(())
            })();
            result.map_err(|error| error.at(function, node_id))?;
        }
        Ok(())
    }

    fn value<'a>(
        &self,
        environment: &'a Environment,
        id: SemanticValueId,
    ) -> Result<&'a Value, EvalError> {
        environment
            .get(&id)
            .ok_or_else(|| EvalError::from("semantic value was used before definition"))
    }

    fn bind_region_parameters(
        &self,
        function: &SemanticFunction,
        region: crate::ids::RegionId,
        captures: &[SemanticValueId],
        child: &mut Environment,
        parent: &Environment,
    ) -> Result<(), EvalError> {
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
    ) -> Result<Value, EvalError> {
        match primitive {
            PrimitiveId::RangeMake => Ok(Value::Range(
                self.value(environment, inputs[0])?.as_nat()?,
                self.value(environment, inputs[1])?.as_nat()?,
            )),
            PrimitiveId::RangeStart | PrimitiveId::RangeEnd => {
                let Value::Range(start, end) = self.value(environment, inputs[0])? else {
                    unreachable!("checked range projection input is not a range")
                };
                Ok(Value::Index(if matches!(primitive, PrimitiveId::RangeStart) {
                    start.clone()
                } else {
                    end.clone()
                }))
            }
            PrimitiveId::Symbolic(expression) => {
                let mut values = assignment.clone();
                for symbol in self.entry.arena().free_symbols((*expression).into()) {
                    if let SymbolKind::RuntimeValue(value) = self.entry.arena().symbol_kind(symbol)
                    {
                        let scalar = self.value(environment, value)?.as_integer()?;
                        // RuntimeValue expressions have the mathematical Int sort,
                        // independently of the producing scalar's storage dtype.
                        values.bind(symbol, SymbolValue::Int(scalar));
                    }
                }
                let value = self
                    .entry
                    .arena()
                    .eval_int(*expression, &values)
                    .map_err(|error| format!("symbolic reference evaluation failed: {error:?}"))?;
                Ok(match function.value(output).ty {
                    SemanticType::Integer => Value::Integer(value),
                    SemanticType::Index { .. } => Value::Index(
                        value.to_biguint().ok_or("negative natural expression result")?,
                    ),
                    SemanticType::Scalar(dtype) => Value::Scalar(
                        word_literal(dtype, &value)?,
                    ),
                    _ => unreachable!("symbolic integer output"),
                })
            }
            PrimitiveId::Constant(constant) => {
                let dtype = scalar_dtype(&function.value(output).ty);
                assert_eq!(
                    constant.dtype(),
                    dtype,
                    "checked constant type differs from its payload"
                );
                Ok(Value::Scalar(*constant))
            }
            PrimitiveId::Unary(operation) => {
                let input = self.value(environment, inputs[0])?;
                if matches!(function.value(output).ty, SemanticType::Integer) {
                    if *operation != crate::syntax::ast::UnaryOp::Neg {
                        return Err("unsupported mathematical integer unary operation".into());
                    }
                    Ok(Value::Integer(-input.as_integer()?))
                } else {
                    Ok(Value::scalar(scalar::unary(*operation, input.as_scalar()?)?))
                }
            }
            PrimitiveId::Binary(operation) => {
                let left = self.value(environment, inputs[0])?;
                let right = self.value(environment, inputs[1])?;
                if matches!(left, Value::Integer(_) | Value::Index(_))
                    || matches!(right, Value::Integer(_) | Value::Index(_)) {
                    use crate::syntax::ast::BinaryOp;
                    let (a, b) = (left.as_integer()?, right.as_integer()?);
                    if let SemanticType::Integer | SemanticType::Index { .. } = function.value(output).ty {
                        let result = match operation {
                            BinaryOp::Add => a + b,
                            BinaryOp::Sub => a - b,
                            BinaryOp::Mul => a * b,
                            BinaryOp::Div | BinaryOp::Rem => {
                                if b.is_zero() {
                                    return Err(crate::reference_math::ScalarFailure::IntegerDivisionByZero.into());
                                }
                                if *operation == BinaryOp::Div { a.div_euclid(&b) } else { a.rem_euclid(&b) }
                            }
                            _ => return Err("unsupported mathematical integer operation".into()),
                        };
                        return Ok(Value::Integer(result));
                    }
                    let comparison = match operation {
                        BinaryOp::Eq => Some(a == b),
                        BinaryOp::Ne => Some(a != b),
                        BinaryOp::Lt => Some(a < b),
                        BinaryOp::Le => Some(a <= b),
                        BinaryOp::Gt => Some(a > b),
                        BinaryOp::Ge => Some(a >= b),
                        _ => None,
                    };
                    if let Some(result) = comparison {
                        Ok(Value::Scalar(ReferenceScalar::Bool(result)))
                    } else {
                        // Authored arithmetic on an index has the checker's
                        // ordinary scalar result dtype; cross that boundary
                        // explicitly instead of turning source I32 into Nat64.
                        let SemanticType::Scalar(dtype) = function.value(output).ty else {
                            unreachable!("checked source index arithmetic has no scalar dtype")
                        };
                        let left = word_literal(dtype, &a)?;
                        let right = word_literal(dtype, &b)?;
                        Ok(Value::Scalar(scalar::binary(
                            *operation,
                            left,
                            right,
                            Some(dtype),
                        )?))
                    }
                } else {
                    Ok(Value::scalar(scalar::binary(
                        *operation,
                        left.as_scalar()?,
                        right.as_scalar()?,
                        Some(scalar_dtype(&function.value(output).ty)),
                    )?))
                }
            }
            PrimitiveId::Cast(dtype) => {
                let value = self.value(environment, inputs[0])?;
                Ok(Value::Scalar(match value {
                    Value::Integer(_) | Value::Index(_) => word_literal(*dtype, &value.as_integer()?)?,
                    _ => scalar::cast(*dtype, value.as_scalar()?),
                }))
            }
            PrimitiveId::Math(operation) => {
                let arguments = inputs
                    .iter()
                    .map(|input| {
                        self.value(environment, *input)?
                            .as_scalar()
                            .map_err(EvalError::from)
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                Ok(Value::scalar(scalar::math(*operation, &arguments)?))
            }
            PrimitiveId::Select => {
                let condition = self.value(environment, inputs[0])?.as_scalar()?.bits() != 0;
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
    ) -> Result<Value, EvalError> {
        let (representation, shape) = self.tensor_type(function, output, assignment)?;
        self.charge_tensor(&shape)?;
        let RepresentationKind::Dense(dtype) = &registry::representation_info(representation).kind
        else {
            unreachable!("computed elementwise result is not dense")
        };
        let count = shape.iter().product();
        let memory = self.reserve_tensor_value(representation, &shape, true)?;
        let _scalars = self.reserve_elements::<ReferenceScalar>(inputs.len())?;
        let mut data = TensorData::uninitialized(representation, shape)?;
        for index in 0..count {
            let mut scalar_inputs = Vec::with_capacity(inputs.len());
            for input in inputs {
                let value = self.value(environment, *input)?;
                let scalar = match value {
                    Value::Scalar(..) => value.clone(),
                    Value::Tensor(tensor) => Value::Scalar(self.read_tensor(tensor, index)?),
                    _ => unreachable!("checked elementwise operand is not numerical"),
                };
                scalar_inputs.push(scalar.as_scalar()?);
            }
            data.write(
                index,
                apply_scalar_primitive(primitive, &scalar_inputs, *dtype)?,
            )?;
        }
        Ok(Value::Tensor(TensorValue::owned(data, memory)))
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
    ) -> Result<Value, EvalError> {
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
        self.charge_tensor(&output_shape)?;
        let output_count: usize = output_shape.iter().product();
        let memory = if matches!(function.value(output).ty, SemanticType::Tensor(_)) {
            Some(self.reserve_tensor_value(registry::dense(output_dtype), &output_shape, true)?)
        } else {
            None
        };
        // The reduction's decoded working results coexist with typed output
        // storage during publication; both payloads belong in the live budget.
        let _results = self.reserve_elements::<ReferenceScalar>(output_count.max(1))?;
        let _coordinates = self.reserve_elements::<usize>(
            input
                .shape
                .len()
                .checked_mul(2)
                .and_then(|n| n.checked_add(output_shape.len() * 2))
                .ok_or("reference rank overflow")?,
        )?;
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
            let working_dtype = accumulator_dtype(operation, input_dtype);
            let mut accumulator = super::tensor::scalar_from_number(
                working_dtype,
                match operation {
                    ReduceOp::Sum => 0.0,
                    ReduceOp::Max | ReduceOp::Argmax => f64::NEG_INFINITY,
                    ReduceOp::Min => f64::INFINITY,
                },
            );
            for coordinate in 0..extent {
                let mut input_coordinate = Vec::with_capacity(input.shape.len());
                input_coordinate.extend_from_slice(&output_coordinate[..axis]);
                input_coordinate.push(coordinate);
                input_coordinate.extend_from_slice(&output_coordinate[axis..]);
                let logical = input_coordinate
                    .iter()
                    .zip(&input_strides)
                    .map(|(coordinate, stride)| coordinate * stride)
                    .sum();
                let value = self.read_tensor(input, logical)?;
                let value = scalar::cast(working_dtype, value);
                match operation {
                    ReduceOp::Sum => {
                        accumulator = scalar::binary(
                            crate::syntax::ast::BinaryOp::Add,
                            accumulator,
                            value,
                            Some(working_dtype),
                        )?
                    }
                    ReduceOp::Max => {
                        accumulator =
                            scalar::math(crate::intrinsics::MathOp::Max, &[accumulator, value])?
                    }
                    ReduceOp::Min => {
                        accumulator =
                            scalar::math(crate::intrinsics::MathOp::Min, &[accumulator, value])?
                    }
                    ReduceOp::Argmax => {
                        if scalar::binary(
                            crate::syntax::ast::BinaryOp::Gt,
                            value,
                            accumulator,
                            None,
                        )?
                        .bits()
                            != 0
                        {
                            accumulator = value;
                            best_index = coordinate;
                        }
                    }
                }
            }
            results.push(if matches!(operation, ReduceOp::Argmax) {
                ReferenceScalar::I32(best_index as i32)
            } else {
                scalar::cast(output_dtype, accumulator)
            });
        }
        if matches!(function.value(output).ty, SemanticType::Tensor(_)) {
            let mut data = TensorData::uninitialized(registry::dense(output_dtype), output_shape)?;
            for (index, value) in results.into_iter().enumerate() {
                data.write(index, value)?;
            }
            Ok(Value::Tensor(TensorValue::owned(data, memory.unwrap())))
        } else {
            Ok(Value::Scalar(results[0]))
        }
    }

    fn tensor_type(
        &self,
        function: &SemanticFunction,
        value: SemanticValueId,
        assignment: &Assignment,
    ) -> Result<(crate::ids::RepresentationId, Vec<usize>), EvalError> {
        let SemanticType::Tensor(tensor) = &function.value(value).ty else {
            unreachable!("checked tensor operation has non-tensor output")
        };
        let _shape = self.reserve_elements::<usize>(tensor.axes.len())?;
        let mut shape = Vec::with_capacity(tensor.axes.len());
        for axis in &tensor.axes {
            let extent = self
                .entry
                .arena()
                .eval_nat(*axis, assignment)
                .map_err(|error| format!("tensor extent evaluation failed: {error:?}"))?;
            shape.push(
                usize::try_from(extent).map_err(|_| "tensor extent exceeds usize".to_owned())?,
            );
        }
        Ok((tensor.representation, shape))
    }

    fn read_tensor(
        &self,
        tensor: &TensorValue,
        logical: usize,
    ) -> Result<ReferenceScalar, EvalError> {
        self.charge_work(1)?;
        let flat = *tensor
            .positions
            .get(logical)
            .ok_or("tensor index outside logical shape")?;
        let info = registry::representation_info(tensor.representation);
        let _decode = self.reserve_elements::<ReferenceScalar>(
            registry::decode_recipe(tensor.representation, info.decoded)
                .map(|recipe| recipe.temporary_count())
                .unwrap_or(0),
        )?;
        match &tensor.backing {
            Backing::Argument(index) => self.tensors[*index].read_scalar(flat).map_err(Into::into),
            Backing::Owned(data) => data.borrow().read_scalar(flat).map_err(Into::into),
        }
    }

    fn write_tensor(
        &mut self,
        tensor: &TensorValue,
        logical: usize,
        value: ReferenceScalar,
    ) -> Result<(), EvalError> {
        self.charge_work(1)?;
        let flat = *tensor
            .positions
            .get(logical)
            .ok_or("tensor index outside logical shape")?;
        match &tensor.backing {
            Backing::Argument(index) => self.tensors[*index].write(flat, value).map_err(Into::into),
            Backing::Owned(data) => data.borrow_mut().write(flat, value).map_err(Into::into),
        }
    }

    fn store_tensor(&mut self, destination: &TensorValue, source: &Value) -> Result<(), EvalError> {
        self.charge_tensor(&destination.shape)?;
        let _snapshot = self.reserve_elements::<ReferenceScalar>(destination.element_count())?;
        let values = match source {
            Value::Scalar(value) => vec![*value; destination.element_count()],
            Value::Tensor(source) => {
                if source.shape != destination.shape {
                    return Err("tensor store shape mismatch".into());
                }
                let mut values = Vec::with_capacity(source.element_count());
                for index in 0..source.element_count() {
                    values.push(self.read_tensor(source, index)?);
                }
                values
            }
            _ => return Err("tensor store source is not numerical".into()),
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
    ) -> Result<usize, EvalError> {
        if indices.len() != tensor.shape.len() {
            unreachable!("checked point access index arity differs from tensor rank")
        }
        let _strides = self.reserve_elements::<usize>(tensor.shape.len())?;
        let strides = row_major(&tensor.shape);
        let mut flat = 0usize;
        for ((index, extent), stride) in indices.iter().zip(tensor.shape.iter()).zip(strides) {
            let index = self.value(environment, *index)?.as_nat()?;
            let index = index.to_usize().ok_or("tensor index exceeds address width")?;
            if index >= *extent {
                return Err("tensor index outside logical shape".into());
            }
            flat += index * stride;
        }
        Ok(flat)
    }

    fn copy_tensor(
        &self,
        function: &SemanticFunction,
        output: SemanticValueId,
        source: &TensorValue,
        assignment: &Assignment,
    ) -> Result<TensorValue, EvalError> {
        let (representation, shape) = self.tensor_type(function, output, assignment)?;
        self.charge_tensor(&shape)?;
        let memory = self.reserve_tensor_value(representation, &shape, true)?;
        if representation == source.representation
            && source
                .positions
                .iter()
                .copied()
                .eq(0..source.element_count())
        {
            // A prefix view can also have indices 0..len. It is an identity
            // of its backing only when the backing geometry matches the copy.
            // Check before cloning so a small view never copies a huge backing.
            let copy = |data: &TensorData| (data.shape() == shape).then(|| data.clone());
            let data = match &source.backing {
                Backing::Argument(index) => copy(&self.tensors[*index]),
                Backing::Owned(data) => copy(&data.borrow()),
            };
            if let Some(data) = data {
                return Ok(TensorValue::owned(data, memory));
            }
        }
        let RepresentationKind::Dense(dtype) = &registry::representation_info(representation).kind
        else {
            return Err("non-dense view copy requires a complete identity packet view".into());
        };
        if representation == source.representation {
            let mut bytes = Vec::with_capacity(
                source
                    .element_count()
                    .checked_mul(dtype.bytes() as usize)
                    .ok_or("tensor size overflow")?,
            );
            for flat in source.positions.iter().copied() {
                self.charge_work(1)?;
                match &source.backing {
                    Backing::Argument(index) => {
                        bytes.extend_from_slice(self.tensors[*index].dense_element_bytes(flat)?)
                    }
                    Backing::Owned(data) => {
                        bytes.extend_from_slice(data.borrow().dense_element_bytes(flat)?)
                    }
                }
            }
            return Ok(TensorValue::owned(
                TensorData::dense_from_bytes(*dtype, shape, bytes)?,
                memory,
            ));
        }
        let mut data = TensorData::uninitialized(representation, shape)?;
        for index in 0..source.element_count() {
            data.write(
                index,
                scalar::cast(*dtype, self.read_tensor(source, index)?),
            )?;
        }
        Ok(TensorValue::owned(data, memory))
    }

    fn apply_view(
        &self,
        transform: &ViewTransform,
        extents: &[SemanticValueId],
        mut base: TensorValue,
        environment: &Environment,
        assignment: &Assignment,
    ) -> Result<TensorValue, EvalError> {
        match transform {
            ViewTransform::Identity => Ok(base),
            ViewTransform::Plane { .. } => {
                Err("raw packed planes have no portable reference value".into())
            }
            ViewTransform::Reshape { axes } => {
                assert_eq!(axes.len(), extents.len(), "checked reshape extent arity changed");
                let memory = std::rc::Rc::new(self.reserve_elements::<usize>(axes.len())?);
                let mut shape = Vec::with_capacity(axes.len());
                for extent in extents {
                    shape.push(self.value(environment, *extent)?.as_nat_usize()?);
                }
                if shape.iter().product::<usize>() != base.element_count() {
                    unreachable!("checked reshape changes element count")
                }
                base.shape = std::rc::Rc::new(shape);
                base.memory.shape = memory;
                Ok(base)
            }
            ViewTransform::Transpose { permutation } => {
                self.charge_tensor(&base.shape)?;
                let shape_memory =
                    std::rc::Rc::new(self.reserve_elements::<usize>(base.shape.len())?);
                let positions_memory =
                    std::rc::Rc::new(self.reserve_elements::<usize>(base.element_count())?);
                let _coordinates = self.reserve_elements::<usize>(
                    base.shape
                        .len()
                        .checked_mul(4)
                        .ok_or("reference rank overflow")?,
                )?;
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
                base.shape = std::rc::Rc::new(new_shape);
                base.positions = std::rc::Rc::new(positions);
                base.memory.shape = shape_memory;
                base.memory.positions = positions_memory;
                Ok(base)
            }
            ViewTransform::Slice { axes } => {
                let _choices = self.reserve_elements::<std::ops::Range<usize>>(base.shape.len())?;
                let _coordinates = self.reserve_elements::<usize>(
                    base.shape
                        .len()
                        .checked_mul(2)
                        .ok_or("reference rank overflow")?,
                )?;
                let shape_memory =
                    std::rc::Rc::new(self.reserve_elements::<usize>(base.shape.len())?);
                let old_shape = base.shape.clone();
                let old_strides = row_major(&old_shape);
                let mut choices = Vec::with_capacity(old_shape.len());
                let mut output_shape = Vec::with_capacity(old_shape.len());
                for (axis, transform) in axes.iter().enumerate() {
                    let extent = old_shape[axis];
                    match transform {
                        crate::entry::SliceAxis::Full => {
                            self.charge_work(extent as u64)?;
                            choices.push(0..extent);
                            output_shape.push(extent);
                        }
                        crate::entry::SliceAxis::Point { value, .. } => {
                            let value = self.scalar_ref(value, environment, assignment)?;
                            choices
                                .push(value..value.checked_add(1).ok_or("slice index overflow")?);
                        }
                        crate::entry::SliceAxis::Range { start, end, .. } => {
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
                            self.charge_work(end.saturating_sub(start) as u64)?;
                            choices.push(start..end);
                            output_shape.push(end - start);
                        }
                    }
                }
                for extent in &old_shape[axes.len()..] {
                    self.charge_work(*extent as u64)?;
                    choices.push(0..*extent);
                    output_shape.push(*extent);
                }
                self.charge_tensor(&output_shape)?;
                let positions_memory = std::rc::Rc::new(
                    self.reserve_elements::<usize>(output_shape.iter().product())?,
                );
                let mut positions = Vec::with_capacity(output_shape.iter().product());
                enumerate_coordinates(
                    &choices,
                    0,
                    &mut Vec::with_capacity(old_shape.len()),
                    &mut |coordinate| {
                        let old_flat = coordinate
                            .iter()
                            .zip(&old_strides)
                            .map(|(a, b)| a * b)
                            .sum::<usize>();
                        positions.push(base.positions[old_flat]);
                    },
                );
                base.shape = std::rc::Rc::new(output_shape);
                base.positions = std::rc::Rc::new(positions);
                base.memory.shape = shape_memory;
                base.memory.positions = positions_memory;
                Ok(base)
            }
        }
    }

    fn scalar_ref(
        &self,
        value: &ScalarRef,
        environment: &Environment,
        assignment: &Assignment,
    ) -> Result<usize, EvalError> {
        match value {
            ScalarRef::Static(value) => self
                .entry
                .arena()
                .eval_nat(*value, assignment)
                .and_then(|value| value.to_usize().ok_or(crate::expr::EvalError::Unrepresentable))
                .map_err(|error| EvalError::from(format!("slice bound failed: {error:?}"))),
            ScalarRef::Value(value) => {
                Ok(self.value(environment, *value)?.as_nat_usize()?)
            }
        }
    }

    fn convert_representation(
        &self,
        conversion: RepresentationConversionId,
        source: &TensorValue,
    ) -> Result<TensorValue, EvalError> {
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
        let memory = self.reserve_tensor_value(conversion.destination, &source.shape, true)?;
        let source_data = match &source.backing {
            Backing::Argument(index) => &self.tensors[*index],
            Backing::Owned(data) => {
                return convert_owned_encoded(
                    conversion.id,
                    source.shape.to_vec(),
                    &data.borrow(),
                    memory,
                )
            }
        };
        convert_owned_encoded(conversion.id, source.shape.to_vec(), source_data, memory)
    }
}

fn convert_owned_encoded(
    conversion: RepresentationConversionId,
    shape: Vec<usize>,
    source: &TensorData,
    memory: super::value::TensorMemory,
) -> Result<TensorValue, EvalError> {
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
                        let value = scalar::cast(plane.storage_dtype, value);
                        bytes.copy_from_slice(&value.bits().to_le_bytes()[..width]);
                    }
                }
            }
        }
    }
    Ok(TensorValue::owned(
        TensorData::encoded(conversion.destination, shape, destination)?,
        memory,
    ))
}

fn eval_repack(expression: &RepackExpr, source: &[u8]) -> ReferenceScalar {
    use crate::syntax::ast::BinaryOp;
    match expression {
        RepackExpr::SourceBits { bit, width } => {
            ReferenceScalar::U32(read_bits(source, *bit as usize, u32::from(*width)))
        }
        RepackExpr::ShiftLeft { value, bits } => scalar::binary(
            BinaryOp::Shl,
            eval_repack(value, source),
            ReferenceScalar::U32(u32::from(*bits)),
            None,
        )
        .unwrap(),
        RepackExpr::BitOr(left, right) => scalar::binary(
            BinaryOp::BitOr,
            eval_repack(left, source),
            eval_repack(right, source),
            None,
        )
        .unwrap(),
        RepackExpr::OffsetI32 { value, offset } => scalar::binary(
            BinaryOp::Add,
            scalar::cast(DType::I32, eval_repack(value, source)),
            ReferenceScalar::I32(*offset),
            None,
        )
        .unwrap(),
        RepackExpr::F16ToF32(value) => scalar::cast(
            DType::F32,
            ReferenceScalar::F16(eval_repack(value, source).bits() as u16),
        ),
        RepackExpr::I32ToF32(value) => scalar::cast(
            DType::F32,
            scalar::cast(DType::I32, eval_repack(value, source)),
        ),
        RepackExpr::MultiplyF32(left, right) => scalar::binary(
            BinaryOp::Mul,
            eval_repack(left, source),
            eval_repack(right, source),
            Some(DType::F32),
        )
        .unwrap(),
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

fn scalar_symbol(value: ReferenceScalar) -> SymbolValue {
    match value {
        ReferenceScalar::F32(bits) => SymbolValue::F32(f32::from_bits(bits)),
        ReferenceScalar::F16(bits) => SymbolValue::F16(bits),
        ReferenceScalar::BF16(bits) => SymbolValue::BF16(bits),
        ReferenceScalar::I32(value) => SymbolValue::I32(value),
        ReferenceScalar::U32(value) => SymbolValue::U32(value),
        ReferenceScalar::Bool(value) => SymbolValue::Bool(value),
    }
}

fn word_literal(dtype: DType, value: &BigInt) -> Result<ReferenceScalar, EvalError> {
    if !dtype.is_int() {
        return Err("mathematical quantity requires an integer word conversion".into());
    }
    let bits = (value & BigInt::from(u32::MAX))
        .to_u32()
        .expect("masked mathematical integer fits one word");
    Ok(ReferenceScalar::from_bits(dtype, bits))
}

fn apply_scalar_primitive(
    primitive: &PrimitiveId,
    arguments: &[ReferenceScalar],
    output: DType,
) -> Result<ReferenceScalar, EvalError> {
    match primitive {
        PrimitiveId::Constant(value) => {
            assert_eq!(
                value.dtype(),
                output,
                "checked constant type differs from its payload"
            );
            Ok(*value)
        }
        PrimitiveId::Unary(operation) => {
            scalar::unary(*operation, arguments[0]).map_err(Into::into)
        }
        PrimitiveId::Binary(operation) => {
            scalar::binary(*operation, arguments[0], arguments[1], Some(output)).map_err(Into::into)
        }
        PrimitiveId::Cast(dtype) => Ok(scalar::cast(*dtype, arguments[0])),
        PrimitiveId::Math(operation) => scalar::math(*operation, arguments).map_err(Into::into),
        PrimitiveId::Select => Ok(if arguments[0].bits() != 0 {
            arguments[1]
        } else {
            arguments[2]
        }),
        PrimitiveId::Decode => Ok(scalar::cast(output, arguments[0])),
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

fn enumerate_coordinates(
    choices: &[std::ops::Range<usize>],
    axis: usize,
    coordinate: &mut Vec<usize>,
    visit: &mut dyn FnMut(&[usize]),
) {
    if axis == choices.len() {
        visit(coordinate);
        return;
    }
    for value in choices[axis].clone() {
        coordinate.push(value);
        enumerate_coordinates(choices, axis + 1, coordinate, visit);
        coordinate.pop();
    }
}
