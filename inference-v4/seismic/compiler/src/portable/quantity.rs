//! Lowering of scalar primitives and quantities: host evaluation, scalar
//! kernels and tensor extents.
use super::*;

impl<'f, 'b, B: seismic_native_target::TargetFamily> Lowerer<'f, 'b, B> {
    pub(super) fn emit_host_scalar(
        &mut self,
        node: NodeId,
        primitive: &PrimitiveId,
        inputs: &[SemanticValueId],
        out: SemanticValueId,
    ) {
        use seismic_lang::syntax::ast::BinaryOp as B;
        // A literal quantity, or one over invocation values only, is already
        // an immutable exact source value. Keep
        // that value as the binding itself, so geometry defined by it cannot
        // acquire an unrelated schedule slot and an artificial entry premise.
        // Public results still use their declared publication destination.
        if matches!(self.function.value(out).ty, SemanticType::Integer)
            && !self.function.results().contains(&out)
        {
            let literal = match primitive {
                PrimitiveId::Symbolic(expression)
                    if invocation_evaluable(self.builder.arena(), (*expression).into()) => Some(*expression),
                PrimitiveId::Constant(ReferenceScalar::I32(value)) => {
                    Some(self.builder.arena().int(i64::from(*value)))
                }
                PrimitiveId::Constant(ReferenceScalar::U32(value)) => {
                    Some(self.builder.arena().int(i64::from(*value)))
                }
                _ => None,
            };
            if let Some(literal) = literal {
                self.values.bind(self.builder.bindings_mut(), out,
                    Bound::Scalar(ScalarBinding::Integer(literal)));
                return;
            }
        }
        let operands = inputs.iter().map(|value| self.bound(*value)).collect::<Vec<_>>();
        let target = self.output_target(out);
        let arena = self.builder.arena();
        let integer = match primitive {
            PrimitiveId::Symbolic(expression) => *expression,
            PrimitiveId::Constant(ReferenceScalar::I32(value)) => arena.int(i64::from(*value)),
            PrimitiveId::Constant(ReferenceScalar::U32(value)) => arena.int(i64::from(*value)),
            PrimitiveId::Unary(ast::UnaryOp::Neg) => {
                let zero = arena.int(0);
                let value = host_integer(arena, &operands[0]);
                arena.int_sub(zero, value)
            }
            PrimitiveId::Binary(op) => {
                let left = host_integer(arena, &operands[0]);
                let right = host_integer(arena, &operands[1]);
                match op {
                    B::Add => arena.int_add(left, right),
                    B::Sub => arena.int_sub(left, right),
                    B::Mul => arena.int_mul(left, right),
                    B::Div => arena.int_div(left, right),
                    B::Rem => arena.int_rem(left, right),
                    B::Eq | B::Ne | B::Lt | B::Le | B::Gt | B::Ge => left,
                    _ => panic!("checked quantity operation has no exact host meaning: {op:?}"),
                }
            }
            PrimitiveId::Cast(_) => host_integer(arena, &operands[0]),
            PrimitiveId::Select => {
                let condition = condition_expr(arena, &self.values.host_conditions, &operands[0]);
                let yes = host_integer(arena, &operands[1]);
                let no = host_integer(arena, &operands[2]);
                arena.int_select(condition, yes, no)
            }
            _ => panic!("checked quantity primitive has no exact host meaning: {primitive:?}"),
        };
        let (value, to) = match &target {
            Bound::Scalar(ScalarBinding::Quantity(slot)) => {
                let value = match slot.kind() {
                    seismic_ir::schedule::HostQuantityKind::Integer => HostValueExpr::Integer(integer),
                    seismic_ir::schedule::HostQuantityKind::Natural => HostValueExpr::Natural(arena.nat_from_int(integer)),
                };
                (value, HostValueDestination::Quantity(*slot))
            }
            Bound::Scalar(ScalarBinding::Published(slot)) => {
                let seismic_ir::repr::ScalarKind::Scalar(dtype) = slot.kind() else {
                    panic!("exact host scalar has a native natural destination")
                };
                let value = match (dtype, primitive) {
                    (DType::Bool, PrimitiveId::Binary(op)) => {
                        let left = host_integer(arena, &operands[0]);
                        let right = host_integer(arena, &operands[1]);
                        let comparison = match op {
                            B::Eq => seismic_lang::expr::CmpOp::Eq,
                            B::Ne => seismic_lang::expr::CmpOp::Ne,
                            B::Lt => seismic_lang::expr::CmpOp::Lt,
                            B::Le => seismic_lang::expr::CmpOp::Le,
                            B::Gt => seismic_lang::expr::CmpOp::Gt,
                            B::Ge => seismic_lang::expr::CmpOp::Ge,
                            _ => panic!("checked quantity Boolean operation is not a comparison"),
                        };
                        HostValueExpr::Bool(arena.int_cmp(comparison, left, right))
                    }
                    (DType::I32 | DType::U32, PrimitiveId::Cast(_)) => {
                        let typed = arena.scalar_integer(
                            seismic_lang::reference_math::ScalarOp::Cast(dtype),
                            &[(dtype, integer)],
                        );
                        HostValueExpr::Word { dtype, value: typed }
                    }
                    (dtype, PrimitiveId::Cast(_)) if dtype.is_float() => HostValueExpr::Float { dtype, value: integer },
                    other => panic!("checked exact-to-word operation lacks its typed conversion: {other:?}"),
                };
                (value, HostValueDestination::Native(*slot))
            }
            _ => panic!("checked host primitive has no scalar publication"),
        };
        let failure = if matches!(primitive, PrimitiveId::Binary(B::Div | B::Rem)) {
            let failure = SourceFailure::at(
                self.function,
                node,
                SourceFailureCause::Scalar(seismic_lang::reference_math::ScalarFailure::IntegerDivisionByZero),
            );
            Some(CheckSite {
                failure,
                path: self.function.name().to_owned(),
                line: self.function.node(node).span().start,
            })
        } else {
            None
        };
        if let (HostValueExpr::Bool(condition), HostValueDestination::Native(slot)) = (value, to) {
            self.values.host_conditions.insert(slot.symbol(), condition);
        }
        self.builder.schedule().evaluate_host(HostEvaluation { value, to, failure });
        self.values.bind(self.builder.bindings_mut(), out, target);
    }

    pub(super) fn emit_scalar(
        &mut self,
        node: NodeId,
        primitive: &PrimitiveId,
        inputs: &[SemanticValueId],
        out: SemanticValueId,
    ) {
        let args = inputs
            .iter()
            .map(|value| self.prepared(&self.bound(*value)))
            .collect::<Vec<_>>();
        let target = self.output_target(out);
        let Bound::Scalar(ScalarBinding::Published(slot)) = target else {
            panic!("scalar primitive output is not scalar")
        };
        let symbolic = if let PrimitiveId::Symbolic(expression) = primitive {
            let (expressions, bindings) = self.builder.expressions_and_bindings();
            Some(capture_expr(
                expressions,
                bindings,
                AnyExpr::Int(*expression),
                &self.values,
            ))
        } else {
            None
        };
        let checks=source_scalar::checks(self.function,node,primitive,inputs);
        let statuses=self.builder.portable_source_statuses(checks.len());
        let mut kernel = self.builder.portable_kernel();
        let status_words=segment_check_statuses(&mut kernel,&checks,&statuses);
        let args = args
            .into_iter()
            .map(|arg| Self::kernel_arg(&mut kernel, arg))
            .collect::<Vec<_>>();
        let alive=kernel.constant(ConstantValue::Bool(true),ValueType::Bool);
        let value = if let Some(expression) = symbolic {
            lower_captured_expr(&mut kernel, &expression)
        } else {
            source_scalar::lower(&mut kernel, self.function, node, &status_words, alive,
                primitive, &args, &self.function.value(out).ty).0
        };
        let storage = slot.kind().value_type();
        let value = if value.ty() != storage {
            kernel.cast(value, storage)
        } else {
            value
        };
        let destination = kernel.result_slot(slot);
        kernel.store_slot(destination,value);
        let kernel = kernel.close();
        self.builder.schedule().launch_sequential(kernel);
        source_scalar::finish(self.builder,self.function.name(),checks,statuses);
        self.values.bind(self.builder.bindings_mut(), out, target);
    }

    pub(super) fn lower_primitive(
        &mut self,
        node: NodeId,
        primitive: &PrimitiveId,
        inputs: &[SemanticValueId],
        output: SemanticValueId,
    ) {
        let resolved;
        let primitive = if let PrimitiveId::Symbolic(expression) = primitive {
            let symbols = self.builder.arena().free_symbols((*expression).into());
            let mut values = Vec::new();
            for symbol in symbols {
                if let SymbolKind::RuntimeValue(value) = self.builder.arena().symbol_kind(symbol) {
                    let bound = self.values.get(self.builder.bindings(), value);
                    let expression = match prepare_scalar(self.builder.arena(), bound.scalar()) {
                        PreparedArg::Index(value) => Some(self.builder.arena().int_from_nat(value)),
                        PreparedArg::Integer(value) => Some(value),
                        PreparedArg::Scalar(symbol, _) => {
                            match self.builder.arena().symbol_sort(symbol) {
                                SymbolSort::Int => Some(self.builder.arena().int_symbol(symbol)),
                                SymbolSort::Nat => {
                                    let value = self.builder.arena().nat_symbol(symbol);
                                    Some(self.builder.arena().int_from_nat(value))
                                }
                                SymbolSort::Scalar(DType::I32) => {
                                    let value = self.builder.arena().scalar_symbol::<seismic_lang::expr::I32>(symbol);
                                    Some(self.builder.arena().int_from_scalar(value))
                                }
                                SymbolSort::Scalar(DType::U32) => {
                                    let value = self.builder.arena().scalar_symbol::<seismic_lang::expr::U32>(symbol);
                                    Some(self.builder.arena().int_from_scalar(value))
                                }
                                SymbolSort::Scalar(_) => None,
                            }
                        }
                    };
                    if let Some(expression) = expression {
                        values.push((value, expression));
                    }
                }
            }
            resolved = PrimitiveId::Symbolic(
                self.builder
                    .arena()
                    .resolve_runtime_values(*expression, &values),
            );
            &resolved
        } else {
            primitive
        };
        if let PrimitiveId::Symbolic(expression) = primitive {
            if matches!(self.function.value(output).ty, SemanticType::Index { .. }) {
                // Index bounds express admissibility, never the scalar value.
                // The checked symbolic operation carries its exact expression.
                let direct = self.builder.arena().nat_from_int(*expression);
                if invocation_evaluable(self.builder.arena(), AnyExpr::Nat(direct)) {
                    self.values.bind(
                        self.builder.bindings_mut(),
                        output,
                        Bound::Scalar(ScalarBinding::Index(direct)),
                    );
                    return;
                }
            }
        }
        match primitive {
            PrimitiveId::TuplePack => panic!("tuple primitive was not canonicalized"),
            PrimitiveId::TupleGet(_) => panic!("tuple projection primitive was not canonicalized"),
            PrimitiveId::RangeMake => {
                let value = Bound::Range {
                    start: Box::new(self.bound(inputs[0])),
                    end: Box::new(self.bound(inputs[1])),
                };
                self.values.bind(self.builder.bindings_mut(), output, value);
            }
            PrimitiveId::RangeStart | PrimitiveId::RangeEnd => {
                let Bound::Range { start, end } = self.bound(inputs[0]) else {
                    panic!("range endpoint input is not a range")
                };
                self.values.bind(
                    self.builder.bindings_mut(),
                    output,
                    if matches!(primitive, PrimitiveId::RangeStart) {
                        *start
                    } else {
                        *end
                    },
                );
            }
            PrimitiveId::TensorAlloc
            | PrimitiveId::Fill(_)
            | PrimitiveId::Copy
            | PrimitiveId::RepresentationConvert(_)
            | PrimitiveId::Transpose
            | PrimitiveId::Reshape
            | PrimitiveId::SliceView { .. }
            | PrimitiveId::ElementRead { .. }
            | PrimitiveId::Extent { .. }
            | PrimitiveId::Atomic { .. }
            | PrimitiveId::Reduce { .. } => {
                panic!("structural primitive survived semantic canonicalization: {primitive:?}")
            }
            _ => {
                if matches!(self.function.value(output).ty, SemanticType::Integer | SemanticType::Index { .. })
                    || inputs.iter().any(|value| matches!(
                        self.function.value(*value).ty,
                        SemanticType::Integer | SemanticType::Index { .. }
                    ))
                {
                    self.emit_host_scalar(node, primitive, inputs, output);
                } else {
                    self.emit_scalar(node, primitive, inputs, output);
                }
            }
        }
    }

    pub(super) fn lower_extent(&mut self, tensor: SemanticValueId, axis: u32, output: SemanticValueId) {
        let view = self.bound(tensor).tensor();
        let value = *view.extents().get(axis as usize)
            .expect("checked extent axis exceeds actual view rank");
        // An axis fixed for the whole invocation is its own exact value, as
        // for an invocation-evaluable index expression. Public results still
        // use their declared publication destination.
        if !self.function.results().contains(&output)
            && invocation_evaluable(self.builder.arena(), AnyExpr::Nat(value))
        {
            let direct = match self.function.value(output).ty {
                SemanticType::Index { .. } => Some(ScalarBinding::Index(value)),
                SemanticType::Integer => Some(ScalarBinding::Integer(self.builder.arena().int_from_nat(value))),
                _ => None,
            };
            if let Some(direct) = direct {
                self.values.bind(self.builder.bindings_mut(), output, Bound::Scalar(direct));
                return;
            }
        }
        let target = self.output_target(output);
        let (value, to) = match &target {
            Bound::Scalar(ScalarBinding::Quantity(slot)) => (
                match slot.kind() {
                    seismic_ir::schedule::HostQuantityKind::Natural => HostValueExpr::Natural(value),
                    seismic_ir::schedule::HostQuantityKind::Integer => {
                        HostValueExpr::Integer(self.builder.arena().int_from_nat(value))
                    }
                },
                HostValueDestination::Quantity(*slot),
            ),
            Bound::Scalar(ScalarBinding::Published(slot)) if
                slot.kind() == seismic_ir::repr::ScalarKind::Scalar(DType::I32) => {
                // `extent` is a source i32 value. The selected view supplies
                // its actual logical axis, and the language cast owns wrapping.
                let exact = self.builder.arena().int_from_nat(value);
                let word = self.builder.arena().scalar_integer(
                    seismic_lang::reference_math::ScalarOp::Cast(DType::I32),
                    &[(DType::I32, exact)],
                );
                (HostValueExpr::Word { dtype: DType::I32, value: word },
                 HostValueDestination::Native(*slot))
            }
            _ => panic!("checked extent result has no typed scalar publication"),
        };
        self.builder.schedule().evaluate_host(HostEvaluation {
            value,
            to,
            failure: None,
        });
        self.values
            .bind(self.builder.bindings_mut(), output, target);
    }
}

/// Only invocation symbols: call values, target constants and decisions.
fn invocation_evaluable(arena: &ExprArena, expression: AnyExpr) -> bool {
    arena.free_symbols(expression).iter().all(|symbol| matches!(
        arena.symbol_kind(*symbol),
        SymbolKind::CallDimension(_) | SymbolKind::CallScalar(_)
            | SymbolKind::TargetConstant(_) | SymbolKind::Decision(_)
    ))
}
