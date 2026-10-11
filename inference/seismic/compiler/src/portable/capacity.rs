//! Successful-path reservation geometry of the actual source tensor product.
//!
//! This derives resource capacity, never the logical extent used by execution.
use super::*;
use seismic_lang::expr::{IntExpr, ScalarOperand};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CapacityPending {
    pub value: SemanticValueId,
    pub reason: CapacityReason,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CapacityReason {
    MissingSourceGeometry,
    UnsupportedShapeExpression,
    UnsupportedReshape,
    UnresolvedBinder,
    RecursiveShape,
    NativeQuantityUnrepresentable,
}

#[derive(Clone, Copy)]
pub(super) struct ShapeInterval {
    lower: IntExpr,
    upper: IntExpr,
}

impl SegmentTensor {
    /// This View tree is constructed by the checked source walker. A raw
    /// PortableTensor slice is deliberately not inspected for source premises.
    fn reservation_axes<B: seismic_native_target::TargetFamily>(
        &self,
        kernel: &mut PortableBuilder<'_, B>,
    ) -> Result<Vec<NatExpr>, CapacityReason> {
        match &self.value {
            TensorDefinitionValue::Physical(storage) => match &storage.origin {
                producer::StorageOrigin::Parameter(contract) => {
                    Ok(contract.view.extents().to_vec())
                }
                producer::StorageOrigin::SourceResult(result) => result.capacity.clone(),
            },
            TensorDefinitionValue::Elementwise { result, .. }
            | TensorDefinitionValue::Reduce { result, .. } => result.capacity.clone(),
            TensorDefinitionValue::View { base, transform } => {
                let base = base.reservation_axes(kernel)?;
                match transform {
                    SegmentViewTransform::Slice(axes) => {
                        let mut result = Vec::new();
                        for (index, selection) in axes.iter().enumerate() {
                            match selection {
                                SegmentSliceAxis::Point(_) => {}
                                SegmentSliceAxis::Full | SegmentSliceAxis::Range { .. } => {
                                    result.push(base[index])
                                }
                            }
                        }
                        result.extend_from_slice(&base[axes.len()..]);
                        Ok(result)
                    }
                    SegmentViewTransform::Transpose(permutation) => {
                        Ok(permutation.iter().map(|i| base[*i as usize]).collect())
                    }
                    SegmentViewTransform::Reshape => self
                        .axes
                        .iter()
                        .map(|value| {
                            kernel
                                .exact_nat(*value)
                                .ok_or(CapacityReason::UnsupportedReshape)
                        })
                        .collect(),
                }
            }
            TensorDefinitionValue::Selected {
                then, otherwise, ..
            } => {
                let then = then.reservation_axes(kernel)?;
                let otherwise = otherwise.reservation_axes(kernel)?;
                assert_eq!(then.len(), otherwise.len());
                Ok(then
                    .into_iter()
                    .zip(otherwise)
                    .map(|(a, b)| kernel.expression_arena().nat_max(a, b))
                    .collect())
            }
        }
    }
}

impl<'s, 'k, 'f, 'r, B: seismic_native_target::TargetFamily> SegmentLowerer<'s, 'k, 'f, 'r, B> {
    fn capacity_substitutions(
        &mut self,
    ) -> std::collections::HashMap<AnyExpr, Result<NatExpr, CapacityReason>> {
        let mut substitutions = std::collections::HashMap::new();
        // Pair each dominating, current-function source tensor with its
        // actual complete product. This includes checked views and helper
        // parameters, so the expression for a slice width retains <= base.
        let products = self
            .values
            .iter()
            .filter_map(|(value, bound)| {
                if value.function() != self.function.id() {
                    return None;
                }
                let (SemanticType::Tensor(formal), SegmentBound::Tensor(actual)) =
                    (&self.function.value(*value).ty, bound)
                else {
                    return None;
                };
                Some((formal.axes.clone(), actual.clone()))
            })
            .collect::<Vec<_>>();
        for (axes, actual) in products {
            let capacity = actual.reservation_axes(self.kernel);
            for (axis, expression) in axes.iter().enumerate() {
                if matches!(
                    self.kernel.expression_arena().view((*expression).into()),
                    NodeView::NatConst(_)
                ) {
                    continue;
                }
                let upper = capacity
                    .as_ref()
                    .map(|axes| axes[axis])
                    .map_err(|reason| *reason);
                substitutions
                    .entry(AnyExpr::Nat(*expression))
                    .and_modify(|old: &mut Result<NatExpr, CapacityReason>| {
                        *old = match (*old, upper) {
                            (Ok(a), Ok(b)) => Ok(self.kernel.expression_arena().nat_min(a, b)),
                            (Ok(a), _) => Ok(a),
                            (_, Ok(b)) => Ok(b),
                            (Err(a), Err(_)) => Err(a),
                        };
                    })
                    .or_insert(upper);
            }
        }
        substitutions
    }

    /// The native quantity path is legal only when every evaluated numeric
    /// intermediate, not merely the final extent, fits the Index domain.
    pub(super) fn shape_native_natural(
        &mut self,
        expression: AnyExpr,
    ) -> Result<bool, CapacityReason> {
        let substitutions = self.capacity_substitutions();
        Inference {
            segment: self,
            substitutions,
            visiting: BTreeSet::new(),
        }
        .native_natural(expression)
    }

    /// A mathematical Integer captured by a participant must fit signed native
    /// storage at every evaluated intermediate. Source word operations retain
    /// their separate modular transfer; this is admission, not a new evaluator.
    pub(super) fn admit_native_quantity(
        &mut self,
        output: SemanticValueId,
        expression: IntExpr,
    ) -> Result<(), CapacityPending> {
        let substitutions = self.capacity_substitutions();
        let mut inference = Inference {
            segment: self,
            substitutions,
            visiting: BTreeSet::new(),
        };
        let admitted = inference.native_integer(expression.into()).unwrap_or(false);
        if admitted {
            let interval =
                inference
                    .expression(expression.into())
                    .map_err(|_| CapacityPending {
                        value: output,
                        reason: CapacityReason::NativeQuantityUnrepresentable,
                    })?;
            inference.segment.quantity_bounds.insert(output, interval);
            Ok(())
        } else {
            Err(CapacityPending {
                value: output,
                reason: CapacityReason::NativeQuantityUnrepresentable,
            })
        }
    }

    pub(super) fn admit_native_integer_value(
        &mut self,
        output: SemanticValueId,
    ) -> Result<(), CapacityPending> {
        let substitutions = self.capacity_substitutions();
        let mut inference = Inference {
            segment: self,
            substitutions,
            visiting: BTreeSet::new(),
        };
        let function = inference.segment.program.function(output.function());
        let operands = match &function.value(output).origin {
            ValueOrigin::Node(node) => match function.node(*node).view() {
                SemanticNodeView::Primitive {
                    primitive: PrimitiveId::Select,
                    inputs,
                    ..
                } => inputs[1..].to_vec(),
                SemanticNodeView::Primitive { inputs, .. } => inputs.to_vec(),
                _ => Vec::new(),
            },
            _ => Vec::new(),
        };
        let operands_fit = operands.into_iter().all(|operand| {
            inference
                .value(operand)
                .and_then(|interval| inference.signed_interval(interval))
                .unwrap_or(false)
        });
        let interval = inference.value(output).ok();
        let admitted = operands_fit
            && interval
                .is_some_and(|interval| inference.signed_interval(interval).unwrap_or(false));
        if admitted {
            inference
                .segment
                .quantity_bounds
                .insert(output, interval.unwrap());
            Ok(())
        } else {
            Err(CapacityPending {
                value: output,
                reason: CapacityReason::NativeQuantityUnrepresentable,
            })
        }
    }

    /// Only already successful source checks refine the continuation. The
    /// check currently being constructed cannot justify its own operands.
    pub(super) fn admit_checked_bounds(&mut self, condition: SemanticValueId) {
        let function = self.program.function(condition.function());
        let ValueOrigin::Node(node) = &function.value(condition).origin else {
            return;
        };
        let SemanticNodeView::Primitive {
            primitive: PrimitiveId::Binary(op),
            inputs,
            ..
        } = function.node(*node).view()
        else {
            return;
        };
        if inputs.len() != 2 {
            return;
        }
        if matches!(op, ast::BinaryOp::And) {
            let inputs = inputs.to_vec();
            self.admit_checked_bounds(inputs[0]);
            self.admit_checked_bounds(inputs[1]);
            return;
        }
        let (left, right, strict) = match op {
            ast::BinaryOp::Le => (inputs[0], inputs[1], false),
            ast::BinaryOp::Lt => (inputs[0], inputs[1], true),
            ast::BinaryOp::Ge => (inputs[1], inputs[0], false),
            ast::BinaryOp::Gt => (inputs[1], inputs[0], true),
            _ => return,
        };
        let substitutions = self.capacity_substitutions();
        let mut inference = Inference {
            segment: self,
            substitutions,
            visiting: BTreeSet::new(),
        };
        let (Ok(a), Ok(b)) = (inference.value(left), inference.value(right)) else {
            return;
        };
        let arena = inference.segment.kernel.expression_arena();
        let one = arena.int(i64::from(strict));
        let a_upper = arena.int_sub(b.upper, one);
        let b_lower = arena.int_add(a.lower, one);
        let a = ShapeInterval {
            lower: a.lower,
            upper: arena.int_min(a.upper, a_upper),
        };
        let b = ShapeInterval {
            lower: arena.int_max(b.lower, b_lower),
            upper: b.upper,
        };
        inference.record_checked_bound(left, a);
        inference.record_checked_bound(right, b);
    }

    pub(super) fn reservation_axes(
        &mut self,
        value: SemanticValueId,
        tensor: &TensorSemantics,
    ) -> Result<Vec<NatExpr>, CapacityPending> {
        let substitutions = self.capacity_substitutions();
        let mut inference = Inference {
            segment: self,
            substitutions,
            visiting: BTreeSet::new(),
        };
        tensor
            .axes
            .iter()
            .map(|axis| {
                let interval = inference
                    .expression((*axis).into())
                    .map_err(|reason| CapacityPending { value, reason })?;
                let arena = inference.segment.kernel.expression_arena();
                let zero = arena.int(0);
                let successful_upper = arena.int_max(zero, interval.upper);
                Ok(arena.nat_from_int(successful_upper))
            })
            .collect()
    }
}

struct Inference<'a, 's, 'k, 'f, 'r, B: seismic_native_target::TargetFamily> {
    segment: &'a mut SegmentLowerer<'s, 'k, 'f, 'r, B>,
    substitutions: std::collections::HashMap<AnyExpr, Result<NatExpr, CapacityReason>>,
    visiting: BTreeSet<SemanticValueId>,
}

impl<B: seismic_native_target::TargetFamily> Inference<'_, '_, '_, '_, '_, B> {
    /// Successful checks constrain the observed source occurrence, not the
    /// inactive filler of a guarded physical publication. Transport only
    /// exact observations of an already evaluated value; arithmetic recipes
    /// and partial or modular conversions are not observation aliases.
    fn record_checked_bound(&mut self, value: SemanticValueId, interval: ShapeInterval) {
        self.segment.quantity_bounds.insert(value, interval);
        let function = self.segment.program.function(value.function());
        let ValueOrigin::Node(node) = &function.value(value).origin else {
            return;
        };
        let SemanticNodeView::Primitive {
            primitive: PrimitiveId::Symbolic(expression),
            ..
        } = function.node(*node).view()
        else {
            return;
        };
        let arena = self.segment.kernel.expression_arena();
        let observed = match arena.view((*expression).into()) {
            NodeView::Unary {
                op: ExprUnary::IntFromScalar | ExprUnary::IntFromNat,
                operand,
            } => operand,
            NodeView::Symbol(_) => (*expression).into(),
            _ => return,
        };
        let NodeView::Symbol(symbol) = arena.view(observed) else {
            return;
        };
        let SymbolKind::RuntimeValue(source) = arena.symbol_kind(symbol) else {
            return;
        };
        if source == value
            || !matches!(
                self.segment.values.get(&source),
                Some(SegmentBound::Scalar(_))
            )
        {
            return;
        }
        let prior = self.value(source).ok();
        let interval = match prior {
            Some(prior) => {
                let arena = self.segment.kernel.expression_arena();
                ShapeInterval {
                    lower: arena.int_max(prior.lower, interval.lower),
                    upper: arena.int_min(prior.upper, interval.upper),
                }
            }
            None => interval,
        };
        self.segment.quantity_bounds.insert(source, interval);
    }

    fn native_integer(&mut self, expression: AnyExpr) -> Result<bool, CapacityReason> {
        let children = match self.segment.kernel.expression_arena().view(expression) {
            NodeView::IntConst(_)
            | NodeView::NatConst(_)
            | NodeView::Symbol(_)
            | NodeView::BoolConst(_) => Vec::new(),
            NodeView::Unary {
                op: ExprUnary::IntFromNat | ExprUnary::IntFromScalar | ExprUnary::NatFromInt,
                operand,
            } => vec![operand],
            NodeView::Binary {
                op:
                    ExprBinary::Add
                    | ExprBinary::Sub
                    | ExprBinary::Mul
                    | ExprBinary::Min
                    | ExprBinary::Max,
                lhs,
                rhs,
            } => vec![lhs, rhs],
            NodeView::Select {
                cond,
                then,
                otherwise,
            } => vec![cond.into(), then, otherwise],
            NodeView::Unary {
                op: ExprUnary::Not,
                operand,
            } => vec![operand],
            NodeView::Binary {
                op: ExprBinary::And | ExprBinary::Or | ExprBinary::Implies | ExprBinary::Iff,
                lhs,
                rhs,
            } => vec![lhs, rhs],
            NodeView::Nary {
                op: NaryOp::All | NaryOp::Any,
                operands,
            } => operands.to_vec(),
            NodeView::Cmp { lhs, rhs, .. } => vec![lhs, rhs],
            NodeView::In { operand, values } => {
                if values.iter().any(|value| i32::try_from(*value).is_err()) {
                    return Ok(false);
                }
                vec![operand]
            }
            NodeView::Nary {
                op: NaryOp::Product,
                operands,
            } => operands.to_vec(),
            NodeView::ScalarConst {
                dtype: DType::I32 | DType::U32 | DType::Bool,
                ..
            } => Vec::new(),
            _ => return Ok(false),
        };
        for child in children {
            if !self.native_integer(child)? {
                return Ok(false);
            }
        }
        let boolean = match expression {
            AnyExpr::Bool(_) => true,
            AnyExpr::Scalar(_) => match self.segment.kernel.expression_arena().view(expression) {
                NodeView::ScalarConst {
                    dtype: DType::Bool, ..
                } => true,
                NodeView::Symbol(symbol) => {
                    self.segment.kernel.expression_arena().symbol_sort(symbol)
                        == SymbolSort::Scalar(DType::Bool)
                }
                _ => false,
            },
            _ => false,
        };
        if boolean {
            return Ok(true);
        }
        let interval = self.expression(expression)?;
        self.signed_interval(interval)
    }

    fn signed_interval(&mut self, interval: ShapeInterval) -> Result<bool, CapacityReason> {
        let arena = self.segment.kernel.expression_arena();
        let min = arena.int(i64::from(i32::MIN));
        let max = arena.int(i64::from(i32::MAX));
        let lower = arena.int_cmp(seismic_lang::expr::CmpOp::Ge, interval.lower, min);
        let upper = arena.int_cmp(seismic_lang::expr::CmpOp::Le, interval.upper, max);
        let fits = arena.all(&[lower, upper]);
        let truth = arena.bool(true);
        Ok(arena
            .eval_bool(fits, &seismic_lang::expr::Assignment::new())
            .ok()
            == Some(true)
            || arena.entails(truth, fits))
    }

    fn native_natural(&mut self, expression: AnyExpr) -> Result<bool, CapacityReason> {
        let (numeric, children) = match self.segment.kernel.expression_arena().view(expression) {
            NodeView::NatConst(_) | NodeView::IntConst(_) => (true, Vec::new()),
            NodeView::BoolConst(_) => (false, Vec::new()),
            NodeView::Symbol(symbol) => {
                match self.segment.kernel.expression_arena().symbol_sort(symbol) {
                    SymbolSort::Nat | SymbolSort::Int => (true, Vec::new()),
                    SymbolSort::Scalar(DType::Bool) => (false, Vec::new()),
                    _ => return Ok(false),
                }
            }
            NodeView::Unary {
                op: ExprUnary::IntFromNat | ExprUnary::NatFromInt,
                operand,
            } => (true, vec![operand]),
            NodeView::Unary {
                op: ExprUnary::Not,
                operand,
            } => (false, vec![operand]),
            NodeView::Binary {
                op:
                    ExprBinary::Add
                    | ExprBinary::Sub
                    | ExprBinary::Mul
                    | ExprBinary::Min
                    | ExprBinary::Max,
                lhs,
                rhs,
            } => (true, vec![lhs, rhs]),
            NodeView::Binary {
                op: ExprBinary::And | ExprBinary::Or | ExprBinary::Implies | ExprBinary::Iff,
                lhs,
                rhs,
            } => (false, vec![lhs, rhs]),
            NodeView::Nary {
                op: NaryOp::Product,
                operands,
            } => (true, operands.to_vec()),
            NodeView::Nary {
                op: NaryOp::All | NaryOp::Any,
                operands,
            } => (false, operands.to_vec()),
            NodeView::Cmp { lhs, rhs, .. } => (false, vec![lhs, rhs]),
            NodeView::In { operand, .. } => (false, vec![operand]),
            NodeView::Select {
                cond,
                then,
                otherwise,
            } => (
                !matches!(expression, AnyExpr::Bool(_)),
                vec![cond.into(), then, otherwise],
            ),
            _ => return Ok(false),
        };
        for child in children {
            if !self.native_natural(child)? {
                return Ok(false);
            }
        }
        if !numeric {
            return Ok(true);
        }
        let interval = self.expression(expression)?;
        let arena = self.segment.kernel.expression_arena();
        let zero = arena.int(0);
        let max = arena.nat(u64::MAX);
        let max = arena.int_from_nat(max);
        let lower = arena.int_cmp(seismic_lang::expr::CmpOp::Ge, interval.lower, zero);
        let upper = arena.int_cmp(seismic_lang::expr::CmpOp::Le, interval.upper, max);
        let fits = arena.all(&[lower, upper]);
        let truth = arena.bool(true);
        Ok(arena
            .eval_bool(fits, &seismic_lang::expr::Assignment::new())
            .ok()
            == Some(true)
            || arena.entails(truth, fits))
    }

    fn exact_nat(&mut self, value: NatExpr) -> ShapeInterval {
        let value = self.segment.kernel.expression_arena().int_from_nat(value);
        ShapeInterval {
            lower: value,
            upper: value,
        }
    }
    fn word(&mut self, dtype: DType) -> Result<ShapeInterval, CapacityReason> {
        word_domain(self.segment.kernel.expression_arena(), dtype)
    }
    fn value(&mut self, value: SemanticValueId) -> Result<ShapeInterval, CapacityReason> {
        // Every successful check contributes a constraint on the same actual
        // scalar occurrence. Meet all active aliases, rather than letting an
        // earlier broad snapshot hide a later reached bound. Exact physical
        // equality also preserves the scalar type; unrelated values do not
        // contribute. The environment owns the lifetime of these constraints.
        let mut intervals = Vec::new();
        if let Some(interval) = self.segment.quantity_bounds.get(&value) {
            intervals.push(*interval);
        }
        if let Some(SegmentBound::Scalar(actual)) = self.segment.values.get(&value) {
            intervals.extend(self.segment.quantity_bounds.iter().filter_map(
                |(other, interval)| match self.segment.values.get(other) {
                    Some(SegmentBound::Scalar(bound)) if other != &value && bound == actual => {
                        Some(*interval)
                    }
                    _ => None,
                },
            ));
        }
        if let Some((first, rest)) = intervals.split_first() {
            let arena = self.segment.kernel.expression_arena();
            let interval = rest.iter().fold(*first, |interval, next| ShapeInterval {
                lower: arena.int_max(interval.lower, next.lower),
                upper: arena.int_min(interval.upper, next.upper),
            });
            return Ok(interval);
        }
        // Helper parameters and captured products preserve the actual scalar
        // occurrence, including its already admitted continuation bounds.
        if let Some(SegmentBound::Scalar(actual)) = self.segment.values.get(&value) {
            // A joined/helper/repeat result may lose its tighter source
            // interval, but an already constructed signed publication still
            // has its supported native domain. Do not reuse an initial carry
            // bound as an invariant of later iterations.
            if matches!(
                self.segment
                    .program
                    .function(value.function())
                    .value(value)
                    .ty,
                SemanticType::Integer
            ) && actual.ty() == ValueType::Scalar(DType::I32)
            {
                return self.word(DType::I32);
            }
        }
        if let Some(SegmentBound::Scalar(scalar)) = self.segment.values.get(&value) {
            if let Some(natural) = self.segment.kernel.exact_nat(*scalar) {
                return Ok(self.exact_nat(natural));
            }
        }
        if !self.visiting.insert(value) {
            return Err(CapacityReason::RecursiveShape);
        }
        let function = self.segment.program.function(value.function());
        let info = function.value(value);
        // An unsuccessful child still ends this traversal. Leaving its marker
        // behind would turn a later independent attempt into a false cycle.
        let result = (|| match &info.origin {
            ValueOrigin::Node(node) => match function.node(*node).view() {
                SemanticNodeView::Extent { tensor, axis, .. } => {
                    let actual = self.segment.tensor(tensor);
                    let capacity = actual.reservation_axes(self.segment.kernel)?;
                    let arena = self.segment.kernel.expression_arena();
                    let interval = ShapeInterval {
                        lower: arena.int(0),
                        upper: arena.int_from_nat(capacity[axis as usize]),
                    };
                    match info.ty {
                        SemanticType::Scalar(dtype) if dtype.is_int() => {
                            // Authored extent getters publish a source word;
                            // their range is modular, unlike internal geometry.
                            self.word_interval(interval, dtype)
                        }
                        SemanticType::Integer | SemanticType::Index { .. } => Ok(interval),
                        _ => Err(CapacityReason::UnsupportedShapeExpression),
                    }
                }
                SemanticNodeView::Primitive {
                    primitive: PrimitiveId::Symbolic(expression),
                    ..
                } => self.expression((*expression).into()),
                SemanticNodeView::Primitive {
                    primitive: PrimitiveId::Constant(value),
                    ..
                } => {
                    let value = match value {
                        ReferenceScalar::I32(value) => i64::from(*value),
                        ReferenceScalar::U32(value) => i64::from(*value),
                        _ => return Err(CapacityReason::UnsupportedShapeExpression),
                    };
                    let value = self.segment.kernel.expression_arena().int(value);
                    Ok(ShapeInterval {
                        lower: value,
                        upper: value,
                    })
                }
                SemanticNodeView::Primitive {
                    primitive, inputs, ..
                } => {
                    let primitive = primitive.clone();
                    let inputs = inputs.to_vec();
                    let output = info.ty.clone();
                    self.primitive_interval(&primitive, &inputs, &output)
                }
                _ => self.value_type(&info.ty),
            },
            _ => self.value_type(&info.ty),
        })();
        self.visiting.remove(&value);
        result
    }
    fn value_type(&mut self, ty: &SemanticType) -> Result<ShapeInterval, CapacityReason> {
        match ty {
            SemanticType::Index { bound } => {
                let bound = self.expression((*bound).into())?;
                let arena = self.segment.kernel.expression_arena();
                let one = arena.int(1);
                let zero = arena.int(0);
                let upper = arena.int_sub(bound.upper, one);
                Ok(ShapeInterval {
                    lower: zero,
                    upper: arena.int_max(upper, zero),
                })
            }
            SemanticType::Scalar(dtype) => self.word(*dtype),
            _ => Err(CapacityReason::UnsupportedShapeExpression),
        }
    }
    fn expression(&mut self, expression: AnyExpr) -> Result<ShapeInterval, CapacityReason> {
        if let Some(upper) = self.substitutions.get(&expression).copied() {
            let upper = upper?;
            let arena = self.segment.kernel.expression_arena();
            return Ok(ShapeInterval {
                lower: arena.int(0),
                upper: arena.int_from_nat(upper),
            });
        }
        let view = self.segment.kernel.expression_arena().view(expression);
        match view {
            NodeView::NatConst(value) => {
                let value = self.segment.kernel.expression_arena().nat(value);
                Ok(self.exact_nat(value))
            }
            NodeView::IntConst(value) => {
                let value = self.segment.kernel.expression_arena().int(value);
                Ok(ShapeInterval {
                    lower: value,
                    upper: value,
                })
            }
            NodeView::ScalarConst { dtype, bits } => {
                let value = match ReferenceScalar::from_bits(dtype, bits) {
                    ReferenceScalar::I32(value) => i64::from(value),
                    ReferenceScalar::U32(value) => i64::from(value),
                    _ => return Err(CapacityReason::UnsupportedShapeExpression),
                };
                let value = self.segment.kernel.expression_arena().int(value);
                Ok(ShapeInterval {
                    lower: value,
                    upper: value,
                })
            }
            NodeView::ScalarInteger {
                operation,
                operands,
            } => {
                let operands = operands.to_vec();
                self.scalar_integer(operation, &operands)
            }
            NodeView::Symbol(symbol) => {
                let arena = self.segment.kernel.expression_arena();
                match arena.symbol_kind(symbol) {
                    SymbolKind::RuntimeValue(value) => self.value(value),
                    SymbolKind::CallDimension(_) | SymbolKind::TargetConstant(_)
                        if arena.symbol_sort(symbol) == SymbolSort::Nat =>
                    {
                        let value = arena.nat_symbol(symbol);
                        Ok(self.exact_nat(value))
                    }
                    SymbolKind::CallScalar(_) => match arena.symbol_sort(symbol) {
                        SymbolSort::Scalar(dtype) => self.word(dtype),
                        SymbolSort::Nat => {
                            let value = arena.nat_symbol(symbol);
                            Ok(self.exact_nat(value))
                        }
                        _ => Err(CapacityReason::UnsupportedShapeExpression),
                    },
                    SymbolKind::LoopBinder(binder) => {
                        let value = self
                            .segment
                            .program
                            .functions()
                            .find_map(|(_, function)| producer::binder_value(function, binder))
                            .filter(|value| self.segment.values.contains_key(value))
                            .ok_or(CapacityReason::UnresolvedBinder)?;
                        self.value(value)
                    }
                    _ => Err(CapacityReason::MissingSourceGeometry),
                }
            }
            NodeView::Unary { op, operand } => {
                let op = op;
                let operand = operand;
                let interval = self.expression(operand)?;
                match op {
                    ExprUnary::IntFromNat | ExprUnary::IntFromScalar => Ok(interval),
                    ExprUnary::NatFromInt => {
                        let arena = self.segment.kernel.expression_arena();
                        let zero = arena.int(0);
                        Ok(ShapeInterval {
                            lower: arena.int_max(interval.lower, zero),
                            upper: arena.int_max(interval.upper, zero),
                        })
                    }
                    _ => Err(CapacityReason::UnsupportedShapeExpression),
                }
            }
            NodeView::Binary { op, lhs, rhs } => {
                let (op, lhs, rhs) = (op, lhs, rhs);
                let a = self.expression(lhs)?;
                let b = self.expression(rhs)?;
                self.binary(op, a, b)
            }
            NodeView::Nary {
                op: NaryOp::Product,
                operands,
            } => {
                let operands = operands.to_vec();
                let one = self.segment.kernel.expression_arena().int(1);
                let mut result = ShapeInterval {
                    lower: one,
                    upper: one,
                };
                for operand in operands {
                    let value = self.expression(operand)?;
                    result = self.binary(ExprBinary::Mul, result, value)?;
                }
                Ok(result)
            }
            NodeView::Select {
                then, otherwise, ..
            } => {
                let (then, otherwise) = (then, otherwise);
                let a = self.expression(then)?;
                let b = self.expression(otherwise)?;
                let arena = self.segment.kernel.expression_arena();
                Ok(ShapeInterval {
                    lower: arena.int_min(a.lower, b.lower),
                    upper: arena.int_max(a.upper, b.upper),
                })
            }
            _ => Err(CapacityReason::UnsupportedShapeExpression),
        }
    }
    /// A source word operation is monotone over these arithmetic intervals
    /// only while its endpoints remain in its declared word domain. Outside
    /// that case retain the real modular type's range, never an unbounded sum.
    fn word_interval(
        &mut self,
        interval: ShapeInterval,
        dtype: DType,
    ) -> Result<ShapeInterval, CapacityReason> {
        word_interval(self.segment.kernel.expression_arena(), interval, dtype)
    }
    fn scalar_integer(
        &mut self,
        operation: seismic_lang::reference_math::ScalarOp,
        operands: &[ScalarOperand],
    ) -> Result<ShapeInterval, CapacityReason> {
        let arena = self.segment.kernel.expression_arena();
        let types = operands
            .iter()
            .map(|operand| operand.dtype(arena))
            .collect::<Vec<_>>();
        let output_dtype = seismic_lang::reference_math::scalar_recipe(operation, &types)
            .output()
            .ty();
        let mut values = Vec::with_capacity(operands.len());
        for operand in operands {
            let dtype = operand.dtype(self.segment.kernel.expression_arena());
            if !dtype.is_int() {
                // The shared recipe owns conversion and payload semantics.
                // Without a source interval for this scalar, retain the full
                // result word domain rather than invent an integer operand.
                return self.word(output_dtype);
            }
            let value = self.expression(operand.expression())?;
            values.push(self.word_interval(value, dtype)?);
        }
        self.operation_interval(operation, output_dtype, &values)
    }

    fn primitive_interval(
        &mut self,
        primitive: &PrimitiveId,
        inputs: &[SemanticValueId],
        output: &SemanticType,
    ) -> Result<ShapeInterval, CapacityReason> {
        if matches!(primitive, PrimitiveId::Copy) && inputs.len() == 1 {
            return self.value(inputs[0]);
        }
        if matches!(output, SemanticType::Integer) {
            let operands = if matches!(primitive, PrimitiveId::Select) {
                &inputs[1..]
            } else {
                inputs
            };
            let values = operands
                .iter()
                .map(|input| self.value(*input))
                .collect::<Result<Vec<_>, _>>()?;
            return match primitive {
                PrimitiveId::Binary(op) => {
                    let op = match op {
                        ast::BinaryOp::Add => ExprBinary::Add,
                        ast::BinaryOp::Sub => ExprBinary::Sub,
                        ast::BinaryOp::Mul => ExprBinary::Mul,
                        _ => return Err(CapacityReason::UnsupportedShapeExpression),
                    };
                    self.binary(op, values[0], values[1])
                }
                PrimitiveId::Unary(ast::UnaryOp::Neg) => {
                    let zero = self.segment.kernel.expression_arena().int(0);
                    self.binary(
                        ExprBinary::Sub,
                        ShapeInterval {
                            lower: zero,
                            upper: zero,
                        },
                        values[0],
                    )
                }
                PrimitiveId::Select => {
                    let arena = self.segment.kernel.expression_arena();
                    Ok(ShapeInterval {
                        lower: arena.int_min(values[0].lower, values[1].lower),
                        upper: arena.int_max(values[0].upper, values[1].upper),
                    })
                }
                _ => Err(CapacityReason::UnsupportedShapeExpression),
            };
        }
        let SemanticType::Scalar(dtype) = output else {
            return self.value_type(output);
        };
        if !dtype.is_int() {
            return self.value_type(output);
        }
        let Some(operation) = seismic_lang::reference_math::scalar_operation(primitive) else {
            return self.word(*dtype);
        };
        let mut values = Vec::with_capacity(inputs.len());
        for input in inputs {
            let ty = self
                .segment
                .program
                .function(input.function())
                .value(*input)
                .ty
                .clone();
            let interval = match ty {
                SemanticType::Scalar(input_dtype) if input_dtype.is_int() => {
                    let interval = self.value(*input)?;
                    self.word_interval(interval, input_dtype)?
                }
                SemanticType::Scalar(_) => return self.word(*dtype),
                _ => self.value(*input)?,
            };
            values.push(interval);
        }
        self.operation_interval(operation, *dtype, &values)
    }

    /// Sound interval transfer for the actual source operation. Operations
    /// without a tighter transfer retain their complete result type domain;
    /// this is capacity information, not an alternate scalar evaluator.
    fn operation_interval(
        &mut self,
        operation: seismic_lang::reference_math::ScalarOp,
        dtype: DType,
        values: &[ShapeInterval],
    ) -> Result<ShapeInterval, CapacityReason> {
        use seismic_lang::reference_math::ScalarOp;
        let value = match operation {
            ScalarOp::Binary(ast::BinaryOp::Add) => {
                self.binary(ExprBinary::Add, values[0], values[1])?
            }
            ScalarOp::Binary(ast::BinaryOp::Sub) => {
                self.binary(ExprBinary::Sub, values[0], values[1])?
            }
            ScalarOp::Binary(ast::BinaryOp::Mul) => {
                self.binary(ExprBinary::Mul, values[0], values[1])?
            }
            ScalarOp::Binary(op @ (ast::BinaryOp::Div | ast::BinaryOp::Rem)) => {
                let domain = self.word(dtype)?;
                let arena = self.segment.kernel.expression_arena();
                let zero = arena.int(0);
                let one = arena.int(1);
                let nonnegative =
                    arena.int_cmp(seismic_lang::expr::CmpOp::Ge, values[0].lower, zero);
                let positive = arena.int_cmp(seismic_lang::expr::CmpOp::Gt, values[1].lower, zero);
                let condition = arena.all(&[nonnegative, positive]);
                let (lower, upper) = if op == ast::BinaryOp::Div {
                    (
                        arena.int_div(values[0].lower, values[1].upper),
                        arena.int_div(values[0].upper, values[1].lower),
                    )
                } else {
                    let headroom = arena.int_sub(values[1].upper, one);
                    (zero, arena.int_min(values[0].upper, headroom))
                };
                ShapeInterval {
                    lower: arena.int_select(condition, lower, domain.lower),
                    upper: arena.int_select(condition, upper, domain.upper),
                }
            }
            ScalarOp::Binary(op @ (ast::BinaryOp::Shl | ast::BinaryOp::Shr)) => {
                let domain = self.word(dtype)?;
                let arena = self.segment.kernel.expression_arena();
                let invocation = seismic_lang::expr::compiled::InvocationValues::new();
                let lower = arena
                    .compile_int(values[1].lower)
                    .evaluate_i64(&invocation)
                    .ok();
                let upper = arena
                    .compile_int(values[1].upper)
                    .evaluate_i64(&invocation)
                    .ok();
                let Some(count) =
                    lower.filter(|count| Some(*count) == upper && (0..32).contains(count))
                else {
                    return Ok(domain);
                };
                let factor = arena.int(1_i64 << count);
                let zero = arena.int(0);
                let condition = arena.int_cmp(seismic_lang::expr::CmpOp::Ge, values[0].lower, zero);
                let (lower, upper) = if op == ast::BinaryOp::Shl {
                    (
                        arena.int_mul(values[0].lower, factor),
                        arena.int_mul(values[0].upper, factor),
                    )
                } else {
                    (
                        arena.int_div(values[0].lower, factor),
                        arena.int_div(values[0].upper, factor),
                    )
                };
                ShapeInterval {
                    lower: arena.int_select(condition, lower, domain.lower),
                    upper: arena.int_select(condition, upper, domain.upper),
                }
            }
            ScalarOp::Math(MathOp::Min) => self.binary(ExprBinary::Min, values[0], values[1])?,
            ScalarOp::Math(MathOp::Max) => self.binary(ExprBinary::Max, values[0], values[1])?,
            ScalarOp::Unary(ast::UnaryOp::Neg) => {
                let arena = self.segment.kernel.expression_arena();
                let zero = arena.int(0);
                ShapeInterval {
                    lower: arena.int_sub(zero, values[0].upper),
                    upper: arena.int_sub(zero, values[0].lower),
                }
            }
            ScalarOp::Math(MathOp::Abs) => {
                let arena = self.segment.kernel.expression_arena();
                let zero = arena.int(0);
                let nonnegative =
                    arena.int_cmp(seismic_lang::expr::CmpOp::Ge, values[0].lower, zero);
                let magnitude = arena.int_sub(zero, values[0].lower);
                let upper = arena.int_max(magnitude, values[0].upper);
                ShapeInterval {
                    lower: arena.int_select(nonnegative, values[0].lower, zero),
                    upper,
                }
            }
            ScalarOp::Cast(DType::I32 | DType::U32) => values[0],
            // Bit operations and payload
            // conversion retain their exact source behavior. Their successful
            // results always lie in the declared word domain.
            _ => return self.word(dtype),
        };
        self.word_interval(value, dtype)
    }
    fn binary(
        &mut self,
        op: ExprBinary,
        a: ShapeInterval,
        b: ShapeInterval,
    ) -> Result<ShapeInterval, CapacityReason> {
        let arena = self.segment.kernel.expression_arena();
        Ok(match op {
            ExprBinary::Add => ShapeInterval {
                lower: arena.int_add(a.lower, b.lower),
                upper: arena.int_add(a.upper, b.upper),
            },
            ExprBinary::Sub => ShapeInterval {
                lower: arena.int_sub(a.lower, b.upper),
                upper: arena.int_sub(a.upper, b.lower),
            },
            ExprBinary::Min => ShapeInterval {
                lower: arena.int_min(a.lower, b.lower),
                upper: arena.int_min(a.upper, b.upper),
            },
            ExprBinary::Max => ShapeInterval {
                lower: arena.int_max(a.lower, b.lower),
                upper: arena.int_max(a.upper, b.upper),
            },
            ExprBinary::Mul => {
                let products = [
                    arena.int_mul(a.lower, b.lower),
                    arena.int_mul(a.lower, b.upper),
                    arena.int_mul(a.upper, b.lower),
                    arena.int_mul(a.upper, b.upper),
                ];
                let mut lower = products[0];
                let mut upper = products[0];
                for value in &products[1..] {
                    lower = arena.int_min(lower, *value);
                    upper = arena.int_max(upper, *value);
                }
                ShapeInterval { lower, upper }
            }
            _ => return Err(CapacityReason::UnsupportedShapeExpression),
        })
    }
}

fn word_domain(arena: &mut ExprArena, dtype: DType) -> Result<ShapeInterval, CapacityReason> {
    let (lower, upper) = match dtype {
        DType::I32 => (i64::from(i32::MIN), i64::from(i32::MAX)),
        DType::U32 => (0, i64::from(u32::MAX)),
        _ => return Err(CapacityReason::UnsupportedShapeExpression),
    };
    Ok(ShapeInterval {
        lower: arena.int(lower),
        upper: arena.int(upper),
    })
}
fn word_interval(
    arena: &mut ExprArena,
    interval: ShapeInterval,
    dtype: DType,
) -> Result<ShapeInterval, CapacityReason> {
    let domain = word_domain(arena, dtype)?;
    let lower = arena.int_cmp(seismic_lang::expr::CmpOp::Ge, interval.lower, domain.lower);
    let upper = arena.int_cmp(seismic_lang::expr::CmpOp::Le, interval.upper, domain.upper);
    let fits = arena.all(&[lower, upper]);
    Ok(ShapeInterval {
        lower: arena.int_select(fits, interval.lower, domain.lower),
        upper: arena.int_select(fits, interval.upper, domain.upper),
    })
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn modular_interval_does_not_publish_unbounded_overflow_or_narrowing() {
        let mut arena = ExprArena::new();
        for (dtype, lower, upper, expected) in [
            (
                DType::U32,
                i64::from(u32::MAX),
                i64::from(u32::MAX) + 3,
                (0, i64::from(u32::MAX)),
            ),
            (
                DType::I32,
                i64::from(i32::MAX),
                i64::from(i32::MAX) + 3,
                (i64::from(i32::MIN), i64::from(i32::MAX)),
            ),
            (DType::U32, -1, 2, (0, i64::from(u32::MAX))),
            (DType::I32, 1, 4, (1, 4)),
        ] {
            let interval = ShapeInterval {
                lower: arena.int(lower),
                upper: arena.int(upper),
            };
            let result = word_interval(&mut arena, interval, dtype).unwrap();
            let assignment = seismic_lang::expr::Assignment::new();
            assert_eq!(
                (
                    arena.eval_int(result.lower, &assignment).unwrap(),
                    arena.eval_int(result.upper, &assignment).unwrap()
                ),
                (expected.0.into(), expected.1.into())
            );
        }
    }
}
