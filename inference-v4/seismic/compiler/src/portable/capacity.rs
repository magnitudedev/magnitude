//! Successful-path reservation geometry of the actual source tensor product.
//!
//! This derives resource capacity, never the logical extent used by execution.
use super::*;
use seismic_lang::expr::IntExpr;

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
}

#[derive(Clone, Copy)]
struct ShapeInterval {
    lower: IntExpr,
    upper: IntExpr,
}

impl SegmentTensor {
    /// This View tree is constructed by the checked source walker. A raw
    /// PortableTensor slice is deliberately not inspected for source premises.
    fn reservation_axes<B: seismic_target::TargetFamily>(
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

impl<'s, 'k, 'f, 'r, B: seismic_target::TargetFamily> SegmentLowerer<'s, 'k, 'f, 'r, B> {
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

struct Inference<'a, 's, 'k, 'f, 'r, B: seismic_target::TargetFamily> {
    segment: &'a mut SegmentLowerer<'s, 'k, 'f, 'r, B>,
    substitutions: std::collections::HashMap<AnyExpr, Result<NatExpr, CapacityReason>>,
    visiting: BTreeSet<SemanticValueId>,
}

impl<B: seismic_target::TargetFamily> Inference<'_, '_, '_, '_, '_, B> {
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
        Ok(arena.eval_bool(fits, &seismic_lang::expr::Assignment::new()).ok() == Some(true)
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
        if !self.visiting.insert(value) {
            return Err(CapacityReason::RecursiveShape);
        }
        let function = self.segment.program.function(value.function());
        let info = function.value(value);
        let result = match &info.origin {
            ValueOrigin::Node(node) => match function.node(*node).view() {
                SemanticNodeView::Extent { tensor, axis, .. } => {
                    let actual = self.segment.tensor(tensor);
                    let capacity = actual.reservation_axes(self.segment.kernel)?;
                    let arena = self.segment.kernel.expression_arena();
                    Ok(ShapeInterval {
                        lower: arena.int(0),
                        upper: arena.int_from_nat(capacity[axis as usize]),
                    })
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
                _ => self.value_type(&info.ty),
            },
            _ => self.value_type(&info.ty),
        };
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
        operands: &[(DType, IntExpr)],
    ) -> Result<ShapeInterval, CapacityReason> {
        use seismic_lang::reference_math::ScalarOp;
        let mut values = Vec::with_capacity(operands.len());
        for (dtype, value) in operands {
            let value = self.expression((*value).into())?;
            values.push(self.word_interval(value, *dtype)?);
        }
        let dtype = match operation {
            ScalarOp::Cast(dtype) => dtype,
            _ => operands[0].0,
        };
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
            ScalarOp::Cast(DType::I32 | DType::U32) => values[0],
            _ => return Err(CapacityReason::UnsupportedShapeExpression),
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
