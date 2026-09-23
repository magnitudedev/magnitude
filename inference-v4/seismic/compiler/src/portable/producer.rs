//! A computed tensor retains its complete result type and the actual operands
//! captured by that type. Physical axes are an instantiation of this owner.
use super::*;

#[derive(Clone, Debug)]
pub(super) struct TensorResult<C> {
    pub(super) value: SemanticValueId,
    pub(super) tensor: TensorSemantics,
    pub(super) capacity: Result<Vec<NatExpr>, capacity::CapacityReason>,
    pub(super) captures: Vec<(ShapeInput, C)>,
    axes: Vec<ResultAxis<C>>,
}

/// An actual bound tensor axis is reused through the complete product. An
/// expression is formed only by the checked native quantity constructor.
#[derive(Clone, Debug)]
enum ResultAxis<C> {
    Bound(C),
    Expression,
    Unavailable(capacity::CapacityReason),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(super) enum ShapeInput {
    Value(SemanticValueId),
    Binder(LoopBinderId),
}
pub(super) fn binder_value(
    function: &SemanticFunction,
    binder: LoopBinderId,
) -> Option<SemanticValueId> {
    function.values().find_map(|(value, info)| {
        let ValueOrigin::RegionParameter(region) = info.origin else {
            return None;
        };
        match function.region(region).kind() {
            RegionKind::LoopBody {
                expression_binder,
                binder_value,
                ..
            } if *expression_binder == binder && *binder_value == value => Some(value),
            _ => None,
        }
    })
}

impl<C> TensorResult<C> {
    fn variables(arena: &ExprArena, axes: &[NatExpr]) -> BTreeSet<ShapeInput> {
        axes.iter()
            .flat_map(|axis| arena.free_symbols((*axis).into()))
            .filter_map(|symbol| match arena.symbol_kind(symbol) {
                SymbolKind::RuntimeValue(value) => Some(ShapeInput::Value(value)),
                SymbolKind::LoopBinder(binder) => Some(ShapeInput::Binder(binder)),
                _ => None,
            })
            .collect()
    }
}

impl<'f, 'b, B: seismic_target::TargetFamily> Lowerer<'f, 'b, B> {
    pub(super) fn bind_tensor_result(
        &mut self,
        value: SemanticValueId,
        tensor: TensorSemantics,
    ) -> TensorResult<PreparedArg> {
        let variables = TensorResult::<PreparedArg>::variables(self.builder.arena(), &tensor.axes);
        let captures = variables
            .iter()
            .copied()
            .map(|input| {
                let value = match input {
                    ShapeInput::Value(value) => value,
                    ShapeInput::Binder(binder) => binder_value(self.function, binder)
                        .expect("shape binder belongs to its checked lexical function"),
                };
                (input, self.prepared(&self.bound(value)))
            })
            .collect();
        let capacity = if variables.is_empty() {
            Ok(tensor.axes.clone())
        } else {
            Err(capacity::CapacityReason::MissingSourceGeometry)
        };
        TensorResult {
            value,
            axes: tensor.axes.iter().map(|_| ResultAxis::Expression).collect(),
            tensor,
            captures,
            capacity,
        }
    }
}

impl<'s, 'k, 'f, 'r, B: seismic_target::TargetFamily> SegmentLowerer<'s, 'k, 'f, 'r, B> {
    pub(super) fn bind_tensor_result(
        &mut self,
        value: SemanticValueId,
        tensor: TensorSemantics,
    ) -> TensorResult<PortableValue> {
        let direct_axes = match self.function.value(value).origin {
            ValueOrigin::Node(node) => match self.function.node(node).view() {
                SemanticNodeView::Alloc { extents, .. }
                | SemanticNodeView::View { extents, transform: ViewTransform::Reshape { .. }, .. } => {
                    Some(extents.iter().map(|id| { let value = self.scalar(*id); portable_index(self.kernel,value) }).collect::<Vec<_>>())
                }
                SemanticNodeView::Fill { like: input, .. }
                | SemanticNodeView::Copy { input, .. }
                | SemanticNodeView::RepresentationConvert { input, .. } => Some(self.tensor(input).axes),
                _ => None,
            },
            _ => None,
        };
        let axes = tensor
            .axes
            .iter()
            .copied()
            .enumerate()
            .map(|(ordinal, axis)| {
                if let Some(actual) = &direct_axes { return ResultAxis::Bound(actual[ordinal]); }
                let bound = self.values.iter().find_map(|(id, bound)| {
                    if id.function() != self.function.id() {
                        return None;
                    }
                    let (SemanticType::Tensor(checked), SegmentBound::Tensor(actual)) =
                        (&self.function.value(*id).ty, bound)
                    else {
                        return None;
                    };
                    checked
                        .axes
                        .iter()
                        .position(|candidate| *candidate == axis)
                        .map(|index| actual.axes[index])
                });
                if let Some(value) = bound {
                    ResultAxis::Bound(value)
                } else if self.shape_native_natural(axis.into()).unwrap_or(false) {
                    ResultAxis::Expression
                } else {
                    ResultAxis::Unavailable(capacity::CapacityReason::UnsupportedShapeExpression)
                }
            })
            .collect::<Vec<_>>();
        let expressions = tensor
            .axes
            .iter()
            .zip(&axes)
            .filter_map(|(expression, axis)| {
                matches!(axis, ResultAxis::Expression).then_some(*expression)
            })
            .collect::<Vec<_>>();
        let variables =
            TensorResult::<PortableValue>::variables(self.kernel.expression_arena(), &expressions);
        let captures = variables
            .into_iter()
            .map(|input| {
                let value = match input {
                    ShapeInput::Value(value) => value,
                    ShapeInput::Binder(binder) => self
                        .program
                        .functions()
                        .find_map(|(_, function)| binder_value(function, binder))
                        .filter(|value| self.values.contains_key(value))
                        .expect("shape binder has an actual dominating participant value"),
                };
                (input, self.scalar(value))
            })
            .collect();
        let capacity = self
            .reservation_axes(value, &tensor)
            .map_err(|pending| pending.reason);
        let mut result = TensorResult {
            value,
            tensor,
            captures,
            capacity,
            axes,
        };
        for index in 0..result.axes.len() {
            if matches!(result.axes[index], ResultAxis::Expression)
                && native_natural_expr(result.capture_axis(self.kernel, result.tensor.axes[index]))
                    .is_none()
            {
                result.axes[index] =
                    ResultAxis::Unavailable(capacity::CapacityReason::UnsupportedShapeExpression);
            }
        }
        result
    }
}

impl TensorResult<PreparedArg> {
    pub(super) fn map_captures(&self, map: &mut impl FnMut(PreparedArg) -> PreparedArg) -> Self {
        Self {
            value: self.value, tensor: self.tensor.clone(), capacity: self.capacity.clone(),
            captures: self.captures.iter().map(|(id,value)|(*id,map(*value))).collect(),
            axes: self.axes.iter().map(|axis|match axis {
                ResultAxis::Bound(value)=>ResultAxis::Bound(map(*value)),
                ResultAxis::Expression=>ResultAxis::Expression,
                ResultAxis::Unavailable(reason)=>ResultAxis::Unavailable(*reason),
            }).collect(),
        }
    }
    pub(super) fn instantiate<B: seismic_target::TargetFamily>(
        &self,
        kernel: &mut PortableBuilder<'_, B>,
    ) -> TensorResult<PortableValue> {
        let mut result = TensorResult {
            value: self.value,
            tensor: self.tensor.clone(),
            capacity: self.capacity.clone(),
            captures: self
                .captures
                .iter()
                .map(|(id, value)| (*id, prepared_kernel_arg(kernel, *value)))
                .collect(),
            axes: Vec::new(),
        };
        for (expression, axis) in result.tensor.axes.iter().copied().zip(&self.axes) {
            let axis = match axis {
                ResultAxis::Bound(value) => ResultAxis::Bound(prepared_kernel_arg(kernel, *value)),
                ResultAxis::Unavailable(reason) => ResultAxis::Unavailable(*reason),
                ResultAxis::Expression => {
                    let captured = result.capture_axis(kernel, expression);
                    if direct_native_natural_expr(&captured)
                        && native_natural_expr(captured).is_some()
                    {
                        ResultAxis::Expression
                    } else {
                        ResultAxis::Unavailable(
                            capacity::CapacityReason::UnsupportedShapeExpression,
                        )
                    }
                }
            };
            result.axes.push(axis);
        }
        if result.capacity.is_err() && result.require_axes().is_ok() {
            result.capacity = result
                .axes(kernel)
                .iter()
                .map(|axis| {
                    kernel
                        .exact_nat(*axis)
                        .ok_or(capacity::CapacityReason::MissingSourceGeometry)
                })
                .collect();
        }
        result
    }
}

impl TensorResult<PortableValue> {
    /// Reserve the checked successful-path envelope, then retain the actual
    /// logical axes in the ordinary zero-origin slice of that backing.
    pub(super) fn snapshot_storage<B: seismic_target::TargetFamily>(
        &self,
        kernel: &mut PortableBuilder<'_, B>,
        axes: &[PortableValue],
    ) -> PortableTensor {
        let storage = kernel.local_tensor(
            LaunchLocalKind::Participant,
            self.tensor.representation,
            self.capacity
                .as_ref()
                .expect("reservation checked before storage construction")
                .clone(),
        );
        let zero = kernel.index_constant(0);
        kernel.tensor_slice(
            storage,
            axes.iter()
                .map(|end| PortableSliceAxis::Range {
                    start: zero,
                    end: *end,
                })
                .collect(),
        )
    }

    fn capture_axis<B: seismic_target::TargetFamily>(
        &self,
        kernel: &mut PortableBuilder<'_, B>,
        axis: NatExpr,
    ) -> CapturedExpr {
        capture_expr_with(kernel.expression_arena(), axis.into(), &mut |_, id| {
            CapturedExpr::Value(
                self.captures
                    .iter()
                    .find(|(input, _)| *input == ShapeInput::Value(id))
                    .expect("shape captured its actual operand")
                    .1,
            )
        })
    }
    pub(super) fn axes<B: seismic_target::TargetFamily>(
        &self,
        kernel: &mut PortableBuilder<'_, B>,
    ) -> Vec<PortableValue> {
        self.tensor
            .axes
            .iter()
            .copied()
            .zip(&self.axes)
            .map(|(expression, axis)| match axis {
                ResultAxis::Bound(value) => *value,
                ResultAxis::Expression => {
                    let captured = native_natural_expr(self.capture_axis(kernel, expression))
                        .expect("quantity constructor selected its actual native expression");
                    let mut binders = self
                        .captures
                        .iter()
                        .filter_map(|(input, value)| match input {
                            ShapeInput::Binder(binder) => Some((*binder, *value)),
                            _ => None,
                        })
                        .collect();
                    let value = lower_captured_expr_with(kernel, &captured, &mut binders);
                    portable_index(kernel, value)
                }
                ResultAxis::Unavailable(_) => {
                    unreachable!("source frame stops before unavailable axis construction")
                }
            })
            .collect()
    }
    pub(super) fn scalar_fields(&self) -> Vec<PortableValue> {
        self.axes
            .iter()
            .filter_map(|axis| match axis {
                ResultAxis::Bound(value) => Some(*value),
                _ => None,
            })
            .chain(self.captures.iter().map(|(_, value)| *value))
            .collect()
    }
    pub(super) fn with_scalar_fields(
        &self,
        fields: &mut impl Iterator<Item = PortableValue>,
    ) -> Self {
        Self {
            value: self.value,
            tensor: self.tensor.clone(),
            capacity: self.capacity.clone(),
            axes: self
                .axes
                .iter()
                .map(|axis| match axis {
                    ResultAxis::Bound(_) => {
                        ResultAxis::Bound(fields.next().expect("bound shape axis"))
                    }
                    ResultAxis::Expression => ResultAxis::Expression,
                    ResultAxis::Unavailable(reason) => ResultAxis::Unavailable(*reason),
                })
                .collect(),
            captures: self
                .captures
                .iter()
                .map(|(id, _)| (*id, fields.next().expect("shape capture")))
                .collect(),
        }
    }
}

/// A physical tensor retains the constructor that owns its source geometry.
/// Parameters reference the actual imported view contract, never a helper's
/// nominal formal axes.
#[derive(Clone, Debug)]
pub(super) struct SegmentStorage {
    pub(super) tensor: PortableTensor,
    pub(super) origin: StorageOrigin,
}
#[derive(Clone, Debug)]
pub(super) enum StorageOrigin {
    Parameter(StoredTensor),
    SourceResult(TensorResult<PortableValue>),
}
impl SegmentStorage {
    pub(super) fn source(result: TensorResult<PortableValue>, tensor: PortableTensor) -> Self {
        Self {
            tensor,
            origin: StorageOrigin::SourceResult(result),
        }
    }
    pub(super) fn scalar_fields(&self) -> Vec<PortableValue> {
        let mut fields = self.tensor.scalar_fields();
        if let StorageOrigin::SourceResult(result) = &self.origin {
            fields.extend(result.scalar_fields());
        }
        fields
    }
    pub(super) fn with_scalar_fields<B: seismic_target::TargetFamily>(
        &self,
        kernel: &mut PortableBuilder<'_, B>,
        fields: &mut impl Iterator<Item = PortableValue>,
    ) -> Self {
        let values = (0..self.tensor.scalar_fields().len())
            .map(|_| fields.next().expect("storage scalar field"))
            .collect::<Vec<_>>();
        let tensor = kernel.tensor_with_scalar_fields(&self.tensor, &values);
        let origin = match &self.origin {
            StorageOrigin::Parameter(contract) => StorageOrigin::Parameter(contract.clone()),
            StorageOrigin::SourceResult(result) => {
                StorageOrigin::SourceResult(result.with_scalar_fields(fields))
            }
        };
        Self { tensor, origin }
    }
}

impl<C> TensorResult<C> {
    /// Every tensor producer needs its actual logical axes immediately, even
    /// if its reservation is deferred until a later materialization.
    pub(super) fn require_axes(&self) -> Result<(), capacity::CapacityPending> {
        for axis in &self.axes {
            if let ResultAxis::Unavailable(reason) = axis {
                return Err(capacity::CapacityPending {
                    value: self.value,
                    reason: *reason,
                });
            }
        }
        Ok(())
    }

    pub(super) fn require_capacity(&self) -> Result<(), capacity::CapacityPending> {
        self.capacity
            .as_ref()
            .map(|_| ())
            .map_err(|reason| capacity::CapacityPending {
                value: self.value,
                reason: *reason,
            })
    }
}
impl<'s, 'k, 'f, 'r, B: seismic_target::TargetFamily> SegmentLowerer<'s, 'k, 'f, 'r, B> {
    /// Build this operation's concrete result before emitting any allocation or
    /// source guard. An unresolved reservation leaves the source cursor here.
    pub(super) fn node_tensor_result(
        &mut self,
        node: NodeId,
    ) -> Result<Option<TensorResult<PortableValue>>, capacity::CapacityPending> {
        let (value, storage) = match self.function.node(node).view() {
            SemanticNodeView::Alloc { output, .. }
            | SemanticNodeView::Fill { output, .. }
            | SemanticNodeView::Copy { output, .. } => (output, true),
            SemanticNodeView::Elementwise {
                output,
                primitive,
                inputs,
            } => (
                output,
                !source_scalar::checks(self.function, node, primitive, inputs).is_empty(),
            ),
            SemanticNodeView::Reduce { output, .. } => (output, false),
            SemanticNodeView::View {
                output,
                transform: ViewTransform::Plane { .. },
                ..
            } => (output, false),
            SemanticNodeView::Store {
                value, destination, ..
            } => {
                let mut destinations = Vec::new();
                snapshot::stored_roots(&self.bound(destination), &mut destinations);
                let source = self.tensor(value);
                (value, source.reads_overlap(self.kernel, &destinations))
            }
            _ => return Ok(None),
        };
        let tensor = match &self.function.value(value).ty {
            SemanticType::Tensor(tensor) => tensor.clone(),
            SemanticType::Scalar(dtype) => TensorSemantics {
                representation: registry::dense(*dtype),
                axes: Vec::new(),
                storage: TensorStorage::Computed,
            },
            _ => unreachable!("tensor constructor result type"),
        };
        let result = self.bind_tensor_result(value, tensor);
        result.require_axes()?;
        if storage {
            result.require_capacity()?;
        }
        Ok(Some(result))
    }
}
