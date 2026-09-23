//! Canonical lowering of checked portable semantics into a closed implementation.
//!
//! This is deliberately a direct constructor, not a recipe or fallback.  It
//! emits a sequential, one-participant schedule whose only choices are the
//! source program's structured control. Backend factories are optimized peers.

pub(crate) mod initialization;
pub(crate) mod construction;
mod streaming;
mod uniformity;
mod products;
mod cohort;
mod source_control;
mod source_scalar;
mod producer;
pub(crate) mod capacity;
#[cfg(test)]
mod capacity_tests;
use producer::{SegmentStorage, StorageOrigin};
mod snapshot;
pub(crate) mod outcome_relation;
mod closed_scalar;
use source_scalar::{SourceChecks, SourceStatuses};
use seismic_lang::failure::{SourceFailure, SourceFailureCause};


use initialization::{StoredView, StoredTensor, StorageContents, CallArgument};
use seismic_lang::initialization::{InitializationArgument, InitializationContext, InitializationState};

use crate::implementation::{
    ImplementationBuilder, PhysicalBinding,
    ScalarBinding, ScalarPublication,
};
use crate::refinement::ConstructedCandidate;
use seismic_ir::kernel::dynamic::{
    PortableBuilder, PortableSliceAxis, PortableTensor, PortableValue,
};
use seismic_ir::kernel::ops::{
    BinaryOp, BitOp, CheckSite, CmpOp, ConstantValue, LogicOp, SegmentLaunchDomain,
    SemanticIntrinsicCall, SemanticIntrinsicOperand, SemanticIntrinsicResult,
    SemanticIntrinsicSink, UnaryOp, ValueType,
};
use seismic_ir::schedule::{AnyScalarSlot, HostEvaluation, HostValueDestination, HostValueExpr, LaunchParticipation};
use seismic_ir::storage::{AnyBufferView, LaunchLocalKind};
use seismic_lang::entry::{
    CheckReason, LoopKind, RegionKind, ScalarRef, SemanticFunction, SemanticNodeView, SemanticType,
    SliceAxis, TensorSemantics, TensorStorage, ValueOrigin, ViewTransform,
};
use seismic_lang::expr::{
    AnyExpr, BinaryOp as ExprBinary, ExprArena, FiniteDomain, FoldOp, IntExpr, LoopBinderId, NaryOp,
    NatExpr, NodeView, SymbolKind, SymbolSort, UnaryOp as ExprUnary,
};
use seismic_lang::ids::{FamilyId, FunctionId, IntrinsicId, NodeId, RegionId, SemanticValueId};
use seismic_lang::intrinsics::{reduce_schema, MathOp, PrimitiveId, ReduceOp};
use seismic_lang::reference_math::ReferenceScalar;
use seismic_lang::registry::{self, RepresentationKind};
use seismic_lang::syntax::ast;
use seismic_lang::types::DType;
use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum SemanticMode {
    Portable,
    PortableParallel,
    AuthoredBackend,
}
/// The required driver uses the same constructor as optional exploration. It
/// supplies the checked reference child at each choice and records that path.
pub(crate) fn construct_general<B: seismic_target::TargetFamily>(
    mut state: construction::SourceConstruction<B>,
    mut context: crate::implementation::ConstructionContext<'_, B>,
) -> Result<(ConstructedCandidate<B>, Vec<(crate::candidate_domain::CallPath, crate::candidate_domain::BodySelection)>), crate::errors::PreparationError> {
    let mut calls = Vec::new();
    loop {
        match state.advance(&mut context) {
            construction::ConstructionStep::Pending(next) => state = next,
            construction::ConstructionStep::Choice(mut next, choice) => {
                let selected = choice.alternatives[0].clone();
                assert_eq!(selected.mapping, crate::candidate_domain::BodyMapping::Sequential,
                    "required construction must select a portable reference child");
                calls.push((choice.path, selected.clone()));
                next.select(selected);
                state = next;
            }
            construction::ConstructionStep::Unresolved(_, reason) => return Err(crate::errors::PreparationError::UnfinishedConstruction(reason)),
            construction::ConstructionStep::Complete(candidate) => return Ok((candidate, calls)),
        }
    }
}

/// Returns the launch-global logical coordinate. Universal portable kernels
/// receive a compiler-owned base argument so later schedule specialization can
/// split one semantic launch without changing the kernel's meaning.
fn logical_global_id<B: seismic_target::TargetFamily>(
    kernel: &mut PortableBuilder<'_, B>,
    compiler_owned: bool,
    extent: NatExpr,
) -> (
    PortableValue,
    Option<seismic_ir::kernel::dynamic::LogicalIndexBinding>,
) {
    if compiler_owned {
        let (logical, binding) = kernel.logical_global_id(extent);
        (logical, Some(binding))
    } else {
        (kernel.global_id(0), None)
    }
}

#[derive(Clone, Debug)]
enum Bound {
    Tensor(TensorRealization),
    Scalar(ScalarBinding),
    Range { start: Box<Bound>, end: Box<Bound> },
    Tuple(Vec<Bound>),
    Unit,
}

struct LoopConstruction {
    schedule: crate::implementation::ScheduleRepeat,
    parent: SemanticBindings,
    initial: Bound,
    start: NatExpr,
    end: NatExpr,
    carries: Vec<seismic_lang::entry::Carry>,
}

/// The value produced by tensor lowering, independent of whether it has storage.
#[derive(Clone, Debug)]
enum TensorRealization {
    Stored(StoredTensor),
    Computed(Arc<StreamTensorPlan>),
}

impl TensorRealization {
    fn stored(&self) -> StoredView {
        match self {
            Self::Stored(value) => value.view.clone(),
            Self::Computed(_) => panic!("addressable use was not materialized by lowering"),
        }
    }
}

#[derive(Clone, Debug)]
enum SegmentBound {
    Tensor(SegmentTensor),
    Scalar(PortableValue),
    Opaque(seismic_ir::kernel::ops::SemanticOpaque),
    Range {
        start: Box<SegmentBound>,
        end: Box<SegmentBound>,
    },
    Tuple(Vec<SegmentBound>),
    Unit,
}

/// One tensor producer definition, instantiated either with construction
/// bindings or with a kernel's physical operands. Instantiation changes the
/// operands, never the operation or its rounding/indexing meaning.
#[derive(Clone, Debug)]
struct TensorDefinition<P, V, I, T, C> {
    axes: Vec<I>,
    value: TensorDefinitionValue<P, V, I, T, C>,
}

#[derive(Clone, Debug)]
enum TensorDefinitionValue<P, V, I, T, C> {
    Physical(P),
    Elementwise {
        primitive: PrimitiveId,
        inputs: Vec<V>,
        result: producer::TensorResult<C>,
    },
    Reduce {
        op: ReduceOp,
        axis: usize,
        input_dtype: DType,
        input: Arc<TensorDefinition<P, V, I, T, C>>,
        result: producer::TensorResult<C>,
    },
    View {
        base: Arc<TensorDefinition<P, V, I, T, C>>,
        transform: T,
    },
    Selected {
        condition: C,
        then: Arc<TensorDefinition<P, V, I, T, C>>,
        otherwise: Arc<TensorDefinition<P, V, I, T, C>>,
    },
}

type SegmentTensor =
    TensorDefinition<SegmentStorage, SegmentBound, PortableValue, SegmentViewTransform, PortableValue>;

#[derive(Clone, Debug)]
enum TensorViewTransform<S> {
    Slice(Vec<TensorSliceAxis<S>>),
    Transpose(Vec<u32>),
    Reshape,
}

#[derive(Clone, Debug)]
enum TensorSliceAxis<S> {
    Full,
    Point(S),
    Range { start: S },
}

type SegmentViewTransform = TensorViewTransform<PortableValue>;
type SegmentSliceAxis = TensorSliceAxis<PortableValue>;

impl SegmentTensor {
    /// A reduction does not modify its captured input versions. Derive its
    /// recurrence scope from those actual addresses and element coordinates.
    fn reduction_uniformity<B: seismic_target::TargetFamily>(
        &self,
        kernel: &PortableBuilder<'_, B>,
        coordinates: registry::IntrinsicUniformity,
    ) -> registry::IntrinsicUniformity {
        let geometry = uniformity::all(self.axes.iter().map(|v| kernel.uniformity(*v)));
        let input = match &self.value {
            TensorDefinitionValue::Physical(tensor) => kernel.tensor_address_uniformity(&tensor.tensor),
            TensorDefinitionValue::Reduce { input, .. } => {
                input.reduction_uniformity(kernel, coordinates)
            }
            TensorDefinitionValue::View { base, transform } => {
                let offsets = match transform {
                    SegmentViewTransform::Slice(axes) => {
                        uniformity::all(axes.iter().map(|axis| match axis {
                            SegmentSliceAxis::Point(v) | SegmentSliceAxis::Range { start: v } => {
                                kernel.uniformity(*v)
                            }
                            SegmentSliceAxis::Full => registry::IntrinsicUniformity::Workgroup,
                        }))
                    }
                    _ => registry::IntrinsicUniformity::Workgroup,
                };
                base.reduction_uniformity(kernel, uniformity::join(coordinates, offsets))
            }
            TensorDefinitionValue::Elementwise { inputs, .. } => {
                uniformity::all(inputs.iter().map(|v| match v {
                    SegmentBound::Scalar(v) => kernel.uniformity(*v),
                    SegmentBound::Tensor(t) => t.reduction_uniformity(kernel, coordinates),
                    _ => unreachable!("elementwise operand kind"),
                }))
            }
            TensorDefinitionValue::Selected { condition, then, otherwise } => uniformity::all([
                kernel.uniformity(*condition),
                then.reduction_uniformity(kernel, coordinates),
                otherwise.reduction_uniformity(kernel, coordinates),
            ]),
        };
        uniformity::all([geometry, input, coordinates])
    }

    fn physical<B: seismic_target::TargetFamily>(
        kernel: &PortableBuilder<'_, B>,
        value: SegmentStorage,
    ) -> Self {
        Self {
            axes: kernel.tensor_extents(&value.tensor).to_vec(),
            value: TensorDefinitionValue::Physical(value),
        }
    }

    fn read<B: seismic_target::TargetFamily>(
        &self,
        kernel: &mut PortableBuilder<'_, B>,
        index: &[PortableValue],
    ) -> PortableValue {
        assert_eq!(
            index.len(),
            self.axes.len(),
            "logical tensor index rank mismatch"
        );
        match &self.value {
            TensorDefinitionValue::Physical(value) => kernel.tensor_read(&value.tensor, index),
            TensorDefinitionValue::Selected { condition, then, otherwise } => {
                kernel.branch(*condition,
                    |kernel| vec![then.read(kernel, index)],
                    |kernel| vec![otherwise.read(kernel, index)])[0]
            }
            TensorDefinitionValue::Elementwise {
                primitive,
                inputs,
                result,
            } => {
                let arguments=segment_elementwise_arguments(kernel,inputs,index);
                let output = SemanticType::Scalar(element_dtype(result.tensor.representation));
                let value = lower_scalar_primitive(kernel, primitive, &arguments, &output);
                let storage_type = value_type(element_dtype(result.tensor.representation));
                if value.ty() == storage_type {
                    value
                } else {
                    kernel.cast(value, storage_type)
                }
            }
            TensorDefinitionValue::Reduce {
                op,
                axis,
                input_dtype,
                input,
                ..
            } => {
                let schema = reduce_schema(*op, *input_dtype);
                let zero = kernel.index_constant(0);
                let end = input.axes[*axis];
                let initial = if matches!(op, ReduceOp::Sum) {
                    zero_of(kernel, value_type(schema.accumulator))
                } else {
                    let first_index = reduction_index(index, *axis, zero);
                    let first = input.read(kernel, &first_index);
                    if first.ty() == value_type(schema.accumulator) {
                        first
                    } else {
                        kernel.cast(first, value_type(schema.accumulator))
                    }
                };
                let start = if matches!(op, ReduceOp::Sum) {
                    zero
                } else {
                    kernel.index_constant(1)
                };
                let carries = if matches!(op, ReduceOp::Argmax) {
                    vec![initial, zero]
                } else {
                    vec![initial]
                };
                let coordinates = uniformity::all(index.iter().map(|v| kernel.uniformity(*v)));
                let recurrence =
                    vec![input.reduction_uniformity(kernel, coordinates); carries.len()];
                let result =
                    kernel.repeat(start, end, carries, &recurrence, |kernel, binder, carry| {
                        let source_index = reduction_index(index, *axis, binder);
                        let value = input.read(kernel, &source_index);
                        let value = if value.ty() == value_type(schema.accumulator) {
                            value
                        } else {
                            kernel.cast(value, value_type(schema.accumulator))
                        };
                        match op {
                            ReduceOp::Sum => vec![kernel.binary(BinaryOp::Add, carry[0], value)],
                            ReduceOp::Max => vec![kernel.binary(BinaryOp::Max, carry[0], value)],
                            ReduceOp::Min => vec![kernel.binary(BinaryOp::Min, carry[0], value)],
                            ReduceOp::Argmax => {
                                let better = kernel.cmp(CmpOp::Gt, value, carry[0]);
                                vec![
                                    kernel.select(better, value, carry[0]),
                                    kernel.select(better, binder, carry[1]),
                                ]
                            }
                        }
                    });
                if matches!(op, ReduceOp::Argmax) {
                    kernel.cast(result[1], ValueType::Scalar(DType::I32))
                } else {
                    result[0]
                }
            }
            TensorDefinitionValue::View { base, transform } => {
                let source = match transform {
                    SegmentViewTransform::Slice(axes) => {
                        let mut output = Vec::with_capacity(axes.len());
                        let mut values = index.iter().copied();
                        for axis in axes {
                            output.push(match axis {
                                SegmentSliceAxis::Point(value) => *value,
                                SegmentSliceAxis::Range { start } => kernel.binary(
                                    BinaryOp::Add,
                                    *start,
                                    values.next().expect("slice mapping rank is closed"),
                                ),
                                SegmentSliceAxis::Full => {
                                    values.next().expect("slice mapping rank is closed")
                                }
                            });
                        }
                        // Omitted trailing source axes are full slices, as in
                        // the semantic view contract (for example x[row]).
                        output.extend(values);
                        assert_eq!(
                            output.len(),
                            base.axes.len(),
                            "slice mapping rank differs from its base"
                        );
                        output
                    }
                    SegmentViewTransform::Transpose(permutation) => {
                        let zero = kernel.index_constant(0);
                        let mut output = vec![zero; permutation.len()];
                        for (axis, source) in permutation.iter().zip(index) {
                            output[*axis as usize] = *source;
                        }
                        output
                    }
                    SegmentViewTransform::Reshape => {
                        let mut linear = kernel.index_constant(0);
                        for (coordinate, extent) in index.iter().zip(&self.axes) {
                            linear = kernel.binary(BinaryOp::Mul, linear, *extent);
                            linear = kernel.binary(BinaryOp::Add, linear, *coordinate);
                        }
                        let zero = kernel.index_constant(0);
                        let mut output = vec![zero; base.axes.len()];
                        for axis in (0..base.axes.len()).rev() {
                            output[axis] = kernel.binary(BinaryOp::Rem, linear, base.axes[axis]);
                            linear = kernel.binary(BinaryOp::Div, linear, base.axes[axis]);
                        }
                        output
                    }
                };
                base.read(kernel, &source)
            }
        }
    }
}

#[derive(Clone, Debug)]
enum SegmentCapture {
    Tensor {
        contract: StoredTensor,
        writable: bool,
    },
    Scalar(PreparedArg),
    Range {
        start: Box<SegmentCapture>,
        end: Box<SegmentCapture>,
    },
    Tuple(Vec<SegmentCapture>),
    Unit,
}

type StreamTensorPlan = TensorDefinition<BindingId, StreamBoundPlan, NatExpr, StreamViewPlan, PreparedArg>;

#[derive(Clone, Debug)]
enum StreamBoundPlan {
    Tensor(Arc<StreamTensorPlan>),
    Scalar(PreparedArg),
}

type StreamViewPlan = TensorViewTransform<PreparedArg>;
type StreamSliceAxisPlan = TensorSliceAxis<PreparedArg>;

fn stored_view_in_kernel<B: seismic_target::TargetFamily>(
    kernel: &mut PortableBuilder<'_, B>, view: &StoredView, writable: bool,
) -> PortableTensor {
    let place = kernel.arg_view(*view.backing(), writable);
    view.map(|_| place, |value| kernel.nat_arg(*value))
}

fn instantiate_stream_tensor<B: seismic_target::TargetFamily>(
    kernel: &mut PortableBuilder<'_, B>,
    plan: &StreamTensorPlan,
    bindings: &BindingArena,
    selections: &BindingSelections,
) -> SegmentTensor {
    let value = match &plan.value {
        TensorDefinitionValue::Physical(binding) => {
            match bindings.get(bindings.selected(*binding, selections)) {
                Bound::Tensor(TensorRealization::Stored(view)) => {
                    let tensor = stored_view_in_kernel(kernel, &view.view, false);
                    TensorDefinitionValue::Physical(SegmentStorage { tensor, origin: StorageOrigin::Parameter(view.clone()) })
                }
                Bound::Tensor(TensorRealization::Computed(plan)) => {
                    return instantiate_stream_tensor(kernel, &plan, bindings, selections);
                }
                _ => panic!("tensor producer capture refers to a non-tensor binding"),
            }
        }
        TensorDefinitionValue::Elementwise {
            primitive,
            inputs,
            result,
        } => TensorDefinitionValue::Elementwise {
            primitive: primitive.clone(),
            inputs: inputs
                .iter()
                .map(|input| match input {
                    StreamBoundPlan::Tensor(tensor) => SegmentBound::Tensor(
                        instantiate_stream_tensor(kernel, tensor, bindings, selections),
                    ),
                    StreamBoundPlan::Scalar(value) => {
                        SegmentBound::Scalar(prepared_kernel_arg(kernel, *value))
                    }
                })
                .collect(),
            result: result.instantiate(kernel),
        },
        TensorDefinitionValue::Reduce {
            op,
            axis,
            input_dtype,
            input,
            result,
        } => TensorDefinitionValue::Reduce {
            op: *op,
            axis: *axis,
            input_dtype: *input_dtype,
            input: Arc::new(instantiate_stream_tensor(
                kernel, input, bindings, selections,
            )),
            result: result.instantiate(kernel),
        },
        TensorDefinitionValue::Selected { condition, then, otherwise } => TensorDefinitionValue::Selected {
            condition: prepared_kernel_arg(kernel, *condition),
            then: Arc::new(instantiate_stream_tensor(kernel, then, bindings, selections)),
            otherwise: Arc::new(instantiate_stream_tensor(kernel, otherwise, bindings, selections)),
        },
        TensorDefinitionValue::View { base, transform } => {
            let base = Arc::new(instantiate_stream_tensor(
                kernel, base, bindings, selections,
            ));
            let transform = match transform {
                StreamViewPlan::Transpose(permutation) => {
                    SegmentViewTransform::Transpose(permutation.clone())
                }
                StreamViewPlan::Reshape => SegmentViewTransform::Reshape,
                StreamViewPlan::Slice(axes) => SegmentViewTransform::Slice(
                    axes.iter()
                        .map(|axis| match axis {
                            StreamSliceAxisPlan::Full => SegmentSliceAxis::Full,
                            StreamSliceAxisPlan::Point(value) => {
                                SegmentSliceAxis::Point(prepared_kernel_arg(kernel, *value))
                            }
                            StreamSliceAxisPlan::Range { start } => SegmentSliceAxis::Range {
                                start: prepared_kernel_arg(kernel, *start),
                            },
                        })
                        .collect(),
                ),
            };
            TensorDefinitionValue::View { base, transform }
        }
    };
    let axes = match &value {
        TensorDefinitionValue::Elementwise { result, .. }
        | TensorDefinitionValue::Reduce { result, .. } => result.axes(kernel),
        _ => plan.axes.iter().map(|axis| kernel.nat_arg(*axis)).collect(),
    };
    SegmentTensor { axes, value }
}

fn prepared_kernel_arg<B: seismic_target::TargetFamily>(
    kernel: &mut PortableBuilder<'_, B>,
    argument: PreparedArg,
) -> PortableValue {
    match argument {
        PreparedArg::Index(value) => kernel.nat_arg(value),
        PreparedArg::Integer(_) => panic!("unbounded Integer cannot enter a fixed native kernel argument"),
        PreparedArg::Scalar(symbol, dtype) => kernel.scalar_arg(symbol, dtype),
    }
}

/// Total construction environment for one checked function. Owner and
/// ordinal checks are concentrated here; lowering sites never join raw IDs
/// against an unqualified map.
#[derive(Clone, Debug)]
struct SemanticBindings {
    function: FunctionId,
    values: Vec<Option<BindingId>>,
    selections: BindingSelections,
    contents: StorageContents,
    binders: Vec<seismic_lang::expr::SymbolId>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct BindingId {
    owner: u64,
    index: usize,
}

/// Physical realizations belong to construction, while lexical environments
/// retain only their actual bound handles. Cloning an environment cannot clone
/// a producer or manufacture a new storage identity.
#[derive(Debug)]
pub(crate) struct BindingArena {
    owner: u64,
    nodes: Vec<BindingValue>,
    parameters: Vec<BindingId>,
    results: Vec<BindingId>,
    parameter_selections: BindingSelections,
    entry_contents: StorageContents,
    entry_binders: Vec<seismic_lang::expr::SymbolId>,
    result_contents: StorageContents,
}

pub(crate) type BindingSelector = seismic_lang::expr::BoolExpr;

pub(crate) type BindingSelections = std::collections::HashMap<BindingSelector, i64>;

pub(crate) fn initialization_context<'a>(
    arena: &'a mut ExprArena,
    selections: &BindingSelections,
    binders: &[seismic_lang::expr::SymbolId],
) -> InitializationContext<'a> {
    let mut context = InitializationContext::new(arena);
    for (selection, value) in selections {
        context.assume(*selection, *value != 0, binders);
    }
    context
}


#[derive(Clone, Debug)]
enum BindingValue {
    Value(Bound),
    Selected {
        selector: BindingSelector,
        options: Vec<(i64, BindingId)>,
    },
}

impl Default for BindingArena {
    fn default() -> Self {
        use std::sync::atomic::{AtomicU64, Ordering};
        static NEXT_OWNER: AtomicU64 = AtomicU64::new(0);
        let owner = NEXT_OWNER
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |id| id.checked_add(1))
            .expect("binding construction owner space exhausted");
        Self {
            owner,
            nodes: Vec::new(),
            parameters: Vec::new(),
            results: Vec::new(),
            parameter_selections: BindingSelections::new(),
            entry_contents: StorageContents::new(),
            entry_binders: Vec::new(),
            result_contents: StorageContents::new(),
        }
    }
}

impl BindingArena {
    fn insert(&mut self, bound: Bound) -> BindingId {
        let id = BindingId {
            owner: self.owner,
            index: self.nodes.len(),
        };
        self.nodes.push(BindingValue::Value(bound));
        id
    }

    fn get(&self, id: BindingId) -> Bound {
        assert_eq!(
            id.owner, self.owner,
            "binding belongs to another construction"
        );
        match &self.nodes[id.index] {
            BindingValue::Value(value) => value.clone(),
            BindingValue::Selected { .. } => {
                panic!("selected binding was not resolved in its construction region")
            }
        }
    }

    fn selected(&self, mut id: BindingId, selections: &BindingSelections) -> BindingId {
        loop {
            assert_eq!(
                id.owner, self.owner,
                "binding belongs to another construction"
            );
            match &self.nodes[id.index] {
                BindingValue::Value(_) => return id,
                BindingValue::Selected { selector, options } => {
                    let value = selections
                        .get(selector)
                        .expect("selected binding has no dominating construction arm");
                    id = options
                        .iter()
                        .find(|(option, _)| option == value)
                        .expect("selected binding option is complete")
                        .1;
                }
            }
        }
    }

    fn unresolved(&self, id: BindingId, selections: &BindingSelections) -> Option<BindingSelector> {
        fn tensor(
            arena: &BindingArena,
            plan: &StreamTensorPlan,
            selections: &BindingSelections,
        ) -> Option<BindingSelector> {
            match &plan.value {
                TensorDefinitionValue::Physical(id) => arena.unresolved(*id, selections),
                TensorDefinitionValue::Elementwise { inputs, .. } => {
                    inputs.iter().find_map(|input| match input {
                        StreamBoundPlan::Tensor(plan) => tensor(arena, plan, selections),
                        StreamBoundPlan::Scalar(_) => None,
                    })
                }
                TensorDefinitionValue::Reduce { input, .. } => tensor(arena, input, selections),
                TensorDefinitionValue::View { base, .. } => tensor(arena, base, selections),
                TensorDefinitionValue::Selected { then, otherwise, .. } => tensor(arena, then, selections).or_else(|| tensor(arena, otherwise, selections)),
            }
        }
        fn bound(
            arena: &BindingArena,
            value: &Bound,
            selections: &BindingSelections,
        ) -> Option<BindingSelector> {
            match value {
                Bound::Tensor(TensorRealization::Computed(plan)) => tensor(arena, plan, selections),
                Bound::Range { start, end } => {
                    bound(arena, start, selections).or_else(|| bound(arena, end, selections))
                }
                Bound::Tuple(values) => values
                    .iter()
                    .find_map(|value| bound(arena, value, selections)),
                _ => None,
            }
        }
        assert_eq!(
            id.owner, self.owner,
            "binding belongs to another construction"
        );
        match &self.nodes[id.index] {
            BindingValue::Value(value) => bound(self, value, selections),
            BindingValue::Selected { selector, options } => match selections.get(selector) {
                Some(value) => self.unresolved(
                    options
                        .iter()
                        .find(|(option, _)| option == value)
                        .expect("selected binding option is complete")
                        .1,
                    selections,
                ),
                None => Some(*selector),
            },
        }
    }

    pub(crate) fn selected_value(
        &mut self,
        selector: BindingSelector,
        options: Vec<(i64, BindingId)>,
    ) -> BindingId {
        let first = options.first().expect("selected value has an option").1;
        for (_, value) in &options {
            assert_eq!(value.owner, self.owner);
        }
        if options.iter().all(|(_, value)| *value == first) {
            return first;
        }
        let id = BindingId {
            owner: self.owner,
            index: self.nodes.len(),
        };
        self.nodes.push(BindingValue::Selected { selector, options });
        id
    }
}

/// Immutable construction products. External parameters remain explicit owner
/// edges so consuming import substitutes the caller's binding, including aliases.
#[derive(Debug, Default)]
pub(crate) struct FrozenBindings {
    arena: BindingArena,
}

impl BindingArena {
    pub(crate) fn physical_binding(
        &self,
        id: BindingId,
        layout: impl FnOnce(AnyBufferView) -> seismic_ir::storage::BufferViewLayout,
    ) -> Option<PhysicalBinding> {
        Some(
            match self.get(self.selected(id, &self.parameter_selections)) {
                Bound::Tensor(TensorRealization::Computed(_)) => return None,
                Bound::Tensor(value) => {
                    let logical = value.stored();
                    let view = *logical.direct_backing()?;
                    PhysicalBinding::View {
                        view,
                        layout: layout(view),
                    }
                }
                Bound::Scalar(value) => PhysicalBinding::Scalar(value),
                Bound::Range { start, end } => PhysicalBinding::Range {
                    start: start.scalar(),
                    end: end.scalar(),
                },
                Bound::Tuple(_) | Bound::Unit => {
                    panic!("call arguments must be normalized value leaves")
                }
            },
        )
    }
    pub(crate) fn parameter_handles(&self) -> &[BindingId] {
        &self.parameters
    }
    pub(crate) fn import_parameters(
        &mut self,
        source: &BindingArena,
        parameters: &[BindingId],
        selections: &BindingSelections,
        contents: &StorageContents,
        binders: &[seismic_lang::expr::SymbolId],
        physical: &mut impl BindingPhysicalImport,
    ) {
        self.parameters = self.import_bindings(source, parameters, &[], physical);
        self.parameter_selections = selections.clone();
        self.entry_contents = contents.clone();
        self.entry_binders = binders.to_vec();
    }
    pub(crate) fn freeze(self) -> FrozenBindings {
        FrozenBindings { arena: self }
    }
}

/// Rebind physical leaves through the existing owning storage/schedule import.
/// The graph importer alone owns semantic node memoization and substitution.
pub(crate) type BindingPath = Vec<(BindingSelector, i64)>;

pub(crate) trait BindingPhysicalImport {
    fn remap_tensor(&mut self, value: &StoredTensor, path: &BindingPath) -> StoredTensor;
    fn remap_slot(&mut self, slot: AnyScalarSlot, path: &BindingPath) -> AnyScalarSlot;
    fn remap_quantity_slot(&mut self, slot: seismic_ir::schedule::HostQuantitySlot, path: &BindingPath) -> seismic_ir::schedule::HostQuantitySlot;
    fn remap_prepared(&mut self, value: PreparedArg, path: &BindingPath) -> PreparedArg;
    fn remap_axis(&mut self, value: NatExpr, path: &BindingPath) -> NatExpr;
    fn remap_selector(&mut self, value: BindingSelector, path: &BindingPath) -> BindingSelector;
}

impl BindingArena {
    /// Visit the storage retained by complete products, including the captured
    /// operands of deferred producers. Historical source associations are not
    /// live products and do not acquire final initialized contents.
    fn visit_stored(&self, products: &[BindingId], visit: &mut impl FnMut(&StoredTensor)) {
        fn tensor(
            value: &StreamTensorPlan,
            arena: &BindingArena,
            seen: &mut std::collections::HashSet<BindingId>,
            visit: &mut impl FnMut(&StoredTensor),
        ) {
            match &value.value {
                TensorDefinitionValue::Physical(id) => node(*id, arena, seen, visit),
                TensorDefinitionValue::Elementwise { inputs, .. } => {
                    for input in inputs {
                        if let StreamBoundPlan::Tensor(input) = input {
                            tensor(input, arena, seen, visit);
                        }
                    }
                }
                TensorDefinitionValue::Reduce { input, .. } => tensor(input, arena, seen, visit),
                TensorDefinitionValue::View { base, .. } => tensor(base, arena, seen, visit),
                TensorDefinitionValue::Selected { then, otherwise, .. } => {
                    tensor(then, arena, seen, visit);
                    tensor(otherwise, arena, seen, visit);
                }
            }
        }
        fn bound(
            value: &Bound,
            arena: &BindingArena,
            seen: &mut std::collections::HashSet<BindingId>,
            visit: &mut impl FnMut(&StoredTensor),
        ) {
            match value {
                Bound::Tensor(TensorRealization::Stored(value)) => visit(value),
                Bound::Tensor(TensorRealization::Computed(value)) => tensor(value, arena, seen, visit),
                Bound::Tuple(values) => {
                    for value in values { bound(value, arena, seen, visit); }
                }
                Bound::Range { start, end } => {
                    bound(start, arena, seen, visit);
                    bound(end, arena, seen, visit);
                }
                Bound::Scalar(_) | Bound::Unit => {}
            }
        }
        fn node(
            id: BindingId,
            arena: &BindingArena,
            seen: &mut std::collections::HashSet<BindingId>,
            visit: &mut impl FnMut(&StoredTensor),
        ) {
            assert_eq!(id.owner, arena.owner);
            if !seen.insert(id) { return; }
            match &arena.nodes[id.index] {
                BindingValue::Value(value) => bound(value, arena, seen, visit),
                BindingValue::Selected { options, .. } => {
                    for (_, id) in options { node(*id, arena, seen, visit); }
                }
            }
        }
        let mut seen = std::collections::HashSet::new();
        for id in products { node(*id, self, &mut seen, visit); }
    }

    pub(crate) fn import_bindings(
        &mut self,
        source: &BindingArena,
        results: &[BindingId],
        parameters: &[(BindingId, BindingId)],
        physical: &mut impl BindingPhysicalImport,
    ) -> Vec<BindingId> {
        let parent = self;
        let mut remap = std::collections::HashMap::new();
        for (source, target) in parameters {
            // Verify the actual caller owns each substituted parameter.
            assert_eq!(
                target.owner, parent.owner,
                "call parameter belongs to another construction"
            );
            remap.insert((*source, Vec::new()), *target);
        }
        fn scalar(
            value: ScalarBinding,
            physical: &mut impl BindingPhysicalImport,
            path: &BindingPath,
        ) -> ScalarBinding {
            match value {
                ScalarBinding::Published(slot) => {
                    ScalarBinding::Published(physical.remap_slot(slot,path))
                }
                ScalarBinding::Quantity(slot) => {
                    ScalarBinding::Quantity(physical.remap_quantity_slot(slot,path))
                }
                ScalarBinding::Value { symbol,dtype } => match physical.remap_prepared(PreparedArg::Scalar(symbol,dtype),path) {
                    PreparedArg::Scalar(symbol,dtype)=>ScalarBinding::Value {symbol,dtype},
                    _=>unreachable!("physical remap preserves scalar category"),
                },
                ScalarBinding::Integer(value) => match physical.remap_prepared(PreparedArg::Integer(value),path) {
                    PreparedArg::Integer(value)=>ScalarBinding::Integer(value),
                    _=>unreachable!("physical remap preserves quantity category"),
                },
                ScalarBinding::Index(value)=>ScalarBinding::Index(physical.remap_axis(value,path)),
            }
        }
        fn tensor(
            source: &StreamTensorPlan,
            child: &BindingArena,
            parent: &mut BindingArena,
            remap: &mut std::collections::HashMap<(BindingId, BindingPath), BindingId>,
            physical: &mut impl BindingPhysicalImport,
            path: &BindingPath,
        ) -> Arc<StreamTensorPlan> {
            let value = match &source.value {
                TensorDefinitionValue::Physical(id) => {
                    TensorDefinitionValue::Physical(node(*id, child, parent, remap, physical, path))
                }
                TensorDefinitionValue::Elementwise {
                    primitive,
                    inputs,
                    result,
                } => TensorDefinitionValue::Elementwise {
                    primitive: primitive.clone(),
                    inputs: inputs
                        .iter()
                        .map(|value| match value {
                            StreamBoundPlan::Tensor(value) => StreamBoundPlan::Tensor(tensor(
                                value, child, parent, remap, physical, path,
                            )),
                            StreamBoundPlan::Scalar(value) => StreamBoundPlan::Scalar(physical.remap_prepared(*value,path)),
                        })
                        .collect(),
                    result: result.map_captures(&mut |value|physical.remap_prepared(value,path)),
                },
                TensorDefinitionValue::Reduce {
                    op,
                    axis,
                    input_dtype,
                    input,
                    result,
                } => TensorDefinitionValue::Reduce {
                    op: *op,
                    axis: *axis,
                    input_dtype: *input_dtype,
                    input: tensor(input, child, parent, remap, physical, path),
                    result: result.map_captures(&mut |value|physical.remap_prepared(value,path)),
                },
                TensorDefinitionValue::View { base, transform } => TensorDefinitionValue::View {
                    base: tensor(base, child, parent, remap, physical, path),
                    transform: match transform {
                        StreamViewPlan::Transpose(permutation)=>StreamViewPlan::Transpose(permutation.clone()),
                        StreamViewPlan::Reshape=>StreamViewPlan::Reshape,
                        StreamViewPlan::Slice(axes)=>StreamViewPlan::Slice(axes.iter().map(|axis|match axis {
                            StreamSliceAxisPlan::Full=>StreamSliceAxisPlan::Full,
                            StreamSliceAxisPlan::Point(value)=>StreamSliceAxisPlan::Point(physical.remap_prepared(*value,path)),
                            StreamSliceAxisPlan::Range {start}=>StreamSliceAxisPlan::Range {start:physical.remap_prepared(*start,path)},
                        }).collect()),
                    },
                },
                TensorDefinitionValue::Selected { condition, then, otherwise } => TensorDefinitionValue::Selected {
                    condition: physical.remap_prepared(*condition,path),
                    then: tensor(then, child, parent, remap, physical, path),
                    otherwise: tensor(otherwise, child, parent, remap, physical, path),
                },
            };
            Arc::new(StreamTensorPlan {
                axes: source.axes.iter().map(|axis|physical.remap_axis(*axis,path)).collect(),
                value,
            })
        }
        fn bound(
            value: Bound,
            child: &BindingArena,
            parent: &mut BindingArena,
            remap: &mut std::collections::HashMap<(BindingId, BindingPath), BindingId>,
            physical: &mut impl BindingPhysicalImport,
            path: &BindingPath,
        ) -> Bound {
            match value {
                Bound::Tensor(TensorRealization::Stored(view)) => {
                    Bound::stored(physical.remap_tensor(&view,path))
                }
                Bound::Tensor(TensorRealization::Computed(plan)) => Bound::Tensor(
                    TensorRealization::Computed(tensor(&plan, child, parent, remap, physical, path)),
                ),
                Bound::Scalar(value) => Bound::Scalar(scalar(value, physical, path)),
                Bound::Range { start, end } => Bound::Range {
                    start: Box::new(bound(*start, child, parent, remap, physical, path)),
                    end: Box::new(bound(*end, child, parent, remap, physical, path)),
                },
                Bound::Tuple(values) => Bound::Tuple(
                    values
                        .into_iter()
                        .map(|value| bound(value, child, parent, remap, physical, path))
                        .collect(),
                ),
                Bound::Unit => Bound::Unit,
            }
        }
        fn node(
            id: BindingId,
            child: &BindingArena,
            parent: &mut BindingArena,
            remap: &mut std::collections::HashMap<(BindingId, BindingPath), BindingId>,
            physical: &mut impl BindingPhysicalImport,
            path: &BindingPath,
        ) -> BindingId {
            if let Some(mapped) = remap.get(&(id, Vec::new())).or_else(||remap.get(&(id,path.clone()))) {
                return *mapped;
            }
            assert_eq!(
                id.owner, child.owner,
                "frozen binding belongs to another construction"
            );
            let mapped = match &child.nodes[id.index] {
                BindingValue::Value(value) => {
                    let value = bound(value.clone(), child, parent, remap, physical, path);
                    parent.insert(value)
                }
                BindingValue::Selected { selector, options } => {
                    let options = options
                        .iter()
                        .map(|(value, binding)| {
                            let mut selected=path.clone(); selected.push((*selector,*value));
                            (*value, node(*binding, child, parent, remap, physical, &selected))
                        })
                        .collect();
                    parent.selected_value(physical.remap_selector(*selector,path), options)
                }
            };
            remap.insert((id,path.clone()), mapped);
            mapped
        }
        results
            .iter()
            .map(|result| node(*result, source, parent, &mut remap, physical, &Vec::new()))
            .collect()
    }
}

impl FrozenBindings {
    pub(crate) fn import_into(
        self,
        parent: &mut BindingArena,
        parameters: &[BindingId],
        physical: &seismic_ir::schedule::ImportedBindings,
        storage: &seismic_ir::storage::TopologyBuilder,
        contents: &mut StorageContents,
    ) -> Vec<BindingId> {
        assert_eq!(
            self.arena.parameters.len(),
            parameters.len(),
            "call binding arity differs"
        );
        struct ScheduleImport<'a> {
            physical: &'a seismic_ir::schedule::ImportedBindings,
            storage: &'a seismic_ir::storage::TopologyBuilder,
        }
        impl BindingPhysicalImport for ScheduleImport<'_> {
            fn remap_tensor(&mut self, value: &StoredTensor, _: &BindingPath) -> StoredTensor {
                let view = value.view.map(|view| self.physical.remap_view(*view), |value| *value);
                let root = self.storage.view_layout(*view.backing()).base;
                StoredTensor { view, root, initialized_view: value.initialized_view.clone() }
            }
            fn remap_slot(&mut self, slot: AnyScalarSlot, _: &BindingPath) -> AnyScalarSlot {
                self.physical.remap_slot(slot)
            }
            fn remap_quantity_slot(&mut self, slot: seismic_ir::schedule::HostQuantitySlot, _: &BindingPath) -> seismic_ir::schedule::HostQuantitySlot {
                self.physical.remap_quantity_slot(slot)
            }
            fn remap_prepared(&mut self, value: PreparedArg, _: &BindingPath) -> PreparedArg { value }
            fn remap_axis(&mut self, value: NatExpr, _: &BindingPath) -> NatExpr { value }
            fn remap_selector(&mut self, value: BindingSelector, _: &BindingPath) -> BindingSelector { value }
        }
        // Only storage retained by escaping products crosses the initialized
        // contents boundary. Parameter effects have already advanced the
        // caller's existing roots through the checked call transfer.
        self.arena.visit_stored(&self.arena.results, &mut |value| {
            let view = physical.remap_view(*value.view.backing());
            let root = storage.view_layout(view).base;
            contents.import_allocation(&self.arena.result_contents, value, root);
        });
        let parameters = self
            .arena
            .parameters
            .iter()
            .copied()
            .zip(parameters.iter().copied())
            .collect::<Vec<_>>();
        // The schedule import already inserts the complete child body,
        // including Unit effects. Only escaping result values need physical
        // remapping into the parent's binding arena.
        parent.import_bindings(
            &self.arena, &self.arena.results, &parameters,
            &mut ScheduleImport { physical, storage },
        )
    }
}

impl SemanticBindings {
    fn new(function: &SemanticFunction) -> Self {
        Self {
            function: function.id(),
            values: vec![None; function.values().count()],
            selections: std::collections::HashMap::new(),
            contents: StorageContents::new(),
            binders: Vec::new(),
        }
    }
    fn slot(&self, value: SemanticValueId) -> usize {
        assert_eq!(
            value.function(),
            self.function,
            "semantic binding belongs to another function"
        );
        let slot = value.index();
        assert!(
            slot < self.values.len(),
            "semantic binding is outside the function arena"
        );
        slot
    }
    fn handle(&self, value: SemanticValueId) -> BindingId {
        self.values[self.slot(value)]
            .expect("checked semantic value was used before its dominating definition")
    }
    fn get(&self, arena: &BindingArena, value: SemanticValueId) -> Bound {
        arena.get(arena.selected(self.handle(value), &self.selections))
    }
    fn bind(&mut self, arena: &mut BindingArena, value: SemanticValueId, bound: Bound) {
        let slot = self.slot(value);
        let binding = arena.insert(bound);
        assert!(self.values[slot].replace(binding).is_none(), "semantic value was bound twice");
    }
    fn rebind(&mut self, arena: &mut BindingArena, value: SemanticValueId, bound: Bound) {
        let slot = self.slot(value);
        // A new residence is scoped to this lexical environment. In particular
        // constructing one branch cannot change its sibling's inherited value.
        let binding = arena.insert(bound);
        self.values[slot] = Some(binding);
    }
    fn contains(&self, value: SemanticValueId) -> bool {
        self.values[self.slot(value)].is_some()
    }
}

impl Bound {
    fn stored(value: StoredTensor) -> Self {
        Self::Tensor(TensorRealization::Stored(value))
    }

    fn scalar(&self) -> ScalarBinding {
        match self {
            Self::Scalar(value) => *value,
            _ => panic!("checked scalar value has no scalar portable binding"),
        }
    }

    fn from_binding(binding: PhysicalBinding, tensor: impl FnOnce(AnyBufferView) -> StoredTensor) -> Self {
        match binding {
            PhysicalBinding::View { view, .. } => Self::stored(tensor(view)),
            PhysicalBinding::Scalar(value) => Self::Scalar(value),
            PhysicalBinding::Range { start, end } => Self::Range {
                start: Box::new(Self::Scalar(start)),
                end: Box::new(Self::Scalar(end)),
            },
        }
    }
    fn tensor(&self) -> StoredView {
        match self {
            Self::Tensor(value) => value.stored(),
            _ => panic!("checked tensor value has no tensor binding"),
        }
    }
}

struct Lowerer<'f, 'b, B: seismic_target::TargetFamily> {
    function: &'f SemanticFunction,
    builder: &'b mut ImplementationBuilder<'f, B>,
    values: SemanticBindings,
    mode: SemanticMode,
    computed_producers: BTreeSet<SemanticValueId>,
}

impl<'f, 'b, B: seismic_target::TargetFamily> Lowerer<'f, 'b, B> {
    fn launch_semantic(
        &mut self,
        kernel: seismic_ir::kernel::KernelId,
        domain: SegmentLaunchDomain,
        logical: Option<seismic_ir::kernel::dynamic::LogicalIndexBinding>,
    ) {
        let descriptor = B::launch_for_participation(self.builder.target().facts(), domain.mode)
            .expect("semantic participation was admitted on an unsupported target");
        self.builder
            .schedule()
            .launch_semantic(kernel, domain, logical, descriptor);
    }

    fn new(
        function: &'f SemanticFunction,
        builder: &'b mut ImplementationBuilder<'f, B>,
        mode: SemanticMode,
    ) -> Self {
        let mut values = SemanticBindings::new(function);
        if builder.bindings().parameters.is_empty() {
            for parameter in function.parameters() {
                let physical = builder.portable_binding(parameter.value);
                let bound = Bound::from_binding(physical, |view| {
                    let layout = builder.portable_layout(view);
                    let root = layout.base;
                    let axes = layout.extents.iter().map(|axis| builder.arena().int_from_nat(*axis)).collect::<Vec<_>>();
                    let view=StoredView::new(view, layout.extents);
                    values.contents.root(&mut InitializationContext::new(builder.arena()), root, view, &axes, InitializationState::full())
                });
                values.bind(builder.bindings_mut(), parameter.value, bound);
                builder
                    .bindings_mut()
                    .parameters
                    .push(values.handle(parameter.value));
            }
            builder.bindings_mut().entry_contents = values.contents.clone();
        } else {
            values.contents = builder.bindings().entry_contents.clone();
            values.binders = builder.bindings().entry_binders.clone();
            assert_eq!(
                function.parameters().len(),
                builder.bindings().parameters.len()
            );
            for (parameter, handle) in function
                .parameters()
                .iter()
                .zip(&builder.bindings().parameters)
            {
                let slot = values.slot(parameter.value);
                values.values[slot] = Some(*handle);
            }
        }
        values.selections = builder.bindings().parameter_selections.clone();
        Self {
            function,
            builder,
            values,
            mode,
            computed_producers: BTreeSet::new(),
        }
    }

    fn independent_domain(&mut self, extent: NatExpr, name: &'static str) -> SegmentLaunchDomain {
        let one = self.builder.arena().nat(1);
        let zero = self.builder.arena().nat(0);
        let workgroup = if self.mode == SemanticMode::Portable {
            // Universal construction is decision-free. A unit workgroup makes
            // every independently defined semantic participant explicit and
            // leaves physical launch chunking to the total-launch certificate.
            one
        } else {
            let domain = FiniteDomain::new(vec![32, 64, 128, 256])
                .expect("canonical independent workgroup domain is non-empty");
            let decision = self.builder.decision(name, domain);
            let selected = self.builder.arena().decision_value(decision);
            self.builder.arena().nat_from_int(selected)
        };
        let groups = self.builder.arena().nat_ceil_div(extent, workgroup);
        let empty = self
            .builder
            .arena()
            .nat_cmp(seismic_lang::expr::CmpOp::Eq, extent, zero);
        SegmentLaunchDomain {
            mode: LaunchParticipation::Independent,
            grid: [groups, one, one],
            workgroup: [workgroup, one, one],
            empty,
            parallel_extent: extent,
        }
    }

    fn may_clobber(&self, id: NodeId) -> bool {
        let node = self.function.node(id);
        if node.events().iter().any(|event| {
            matches!(
                event.kind(),
                seismic_lang::entry::SemanticEventKind::Write(_)
                    | seismic_lang::entry::SemanticEventKind::AtomicRmw { .. }
            )
        }) {
            return true;
        }
        match node.view() {
            SemanticNodeView::Call { family, .. } => self.builder.call_can_write(family),
            SemanticNodeView::If {
                then, otherwise, ..
            } => [then, otherwise].into_iter().any(|region| {
                self.function
                    .nodes(region)
                    .any(|(id, _)| self.may_clobber(id))
            }),
            SemanticNodeView::Loop { body, .. } => self
                .function
                .nodes(body)
                .any(|(id, _)| self.may_clobber(id)),
            _ => false,
        }
    }

    fn node_selection(&self, id: NodeId) -> Option<BindingSelector> {
        if self.may_clobber(id) {
            if let Some(selector) = self.values.values.iter().flatten().find_map(|binding| self.builder.bindings().unresolved(*binding, &self.values.selections)) {
                return Some(selector);
            }
        }
        self.function.node(id).dependencies().into_iter().find_map(|value| {
            self.values.contains(value).then(|| self.builder.bindings().unresolved(self.values.handle(value), &self.values.selections)).flatten()
        })
    }

    fn begin_selected_binding(&mut self, selector: BindingSelector) -> construction::SelectedConstruction {
        let inherited = self.values.clone();
        let (selected, schedule) = self.builder.begin_source_selection(selector);
        self.values.selections.insert(selector, selected);
        construction::SelectedConstruction { selector, schedule, inherited, selected, outcomes: Vec::new() }
    }

    fn next_selected_binding(&mut self, mut progress: construction::SelectedConstruction) -> Option<construction::SelectedConstruction> {
        progress.outcomes.push((progress.selected, self.values.clone()));
        if let Some(selected) = self.builder.next_source_branch(&mut progress.schedule) {
            progress.selected = selected;
            self.values = progress.inherited.clone();
            self.values.selections.insert(progress.selector, selected);
            return Some(progress);
        }
        self.builder.finish_source_branch(progress.schedule);
        self.values = progress.inherited;
        let selector = progress.selector;
        let outcomes = progress.outcomes;
        self.values.contents.branch(
            &mut InitializationContext::new(self.builder.arena()), selector, &self.values.binders,
            &outcomes[0].1.contents, &outcomes[1].1.contents,
        );
        for (source, _) in self.function.values() {
            let slot = self.values.slot(source);
            let mut options = Vec::new();
            let mut absent = 0;
            for (option, environment) in &outcomes {
                match environment.values[slot] {
                    Some(value) => options.push((*option, value)),
                    None => absent += 1,
                }
            }
            self.values.values[slot] = if absent == outcomes.len() {
                None
            } else {
                assert_eq!(
                    absent, 0,
                    "selected operation defines inconsistent semantic outputs"
                );
                Some(
                    self.builder
                        .bindings_mut()
                        .selected_value(selector, options),
                )
            };
        }
        None
    }

    fn snapshot_before_write(&mut self, id: NodeId) {
        if self.may_clobber(id) {
            let snapshots = self
                .function
                .values()
                .filter_map(|(value, _)| {
                    if !self.values.contains(value) {
                        return None;
                    }
                    matches!(
                        self.bound(value),
                        Bound::Tensor(TensorRealization::Computed(_))
                    )
                    .then_some(value)
                })
                .collect::<Vec<_>>();
            for snapshot in snapshots {
                self.materialize_tensor(snapshot);
            }
        }
    }

    fn pending_representation_view(&self, node: NodeId) -> Option<(SemanticValueId, seismic_lang::ids::RepresentationId)> {
        let SemanticNodeView::View { base, transform: ViewTransform::Slice { .. }, output, .. } = self.function.node(node).view() else { return None; };
        if !matches!(self.bound(base), Bound::Tensor(TensorRealization::Stored(_))) { return None; }
        let SemanticType::Tensor(tensor) = &self.function.value(base).ty else { unreachable!() };
        (!matches!(registry::representation_info(tensor.representation).kind, RepresentationKind::Dense(_)))
            .then_some((output, tensor.representation))
    }

    fn lower_node_resolved(&mut self, id: NodeId) {
        self.snapshot_before_write(id);
        let node = self.function.node(id);
        let computed_value = match node.view() {
            SemanticNodeView::Elementwise { .. } => true,
            SemanticNodeView::View { base, .. } => matches!(self.bound(base), Bound::Tensor(TensorRealization::Computed(_))),
            _ => false,
        };
        if computed_value && matches!(node.view(),SemanticNodeView::Elementwise{output,..}|SemanticNodeView::View{output,..} if self.computed_producers.contains(&output))
        {
            let output = match node.view() {
                SemanticNodeView::Elementwise { output, .. }
                | SemanticNodeView::View { output, .. } => output,
                _ => unreachable!(),
            };
            let plan = self.computed_tensor(output);
            self.values.bind(
                self.builder.bindings_mut(),
                output,
                Bound::Tensor(TensorRealization::Computed(plan)),
            );
            return;
        }
        if !matches!(
            node.view(),
            SemanticNodeView::Reduce { .. } | SemanticNodeView::Call { .. }
        ) {
            for dependency in node.dependencies() {
                self.materialize_tensor(dependency);
            }
        }
        match node.view() {
            SemanticNodeView::Primitive {
                primitive,
                inputs,
                output,
            } => self.lower_primitive(id, primitive, inputs, output),
            SemanticNodeView::Intrinsic {
                intrinsic,
                inputs,
                output,
            } => self.lower_intrinsic(intrinsic, inputs, output),
            SemanticNodeView::Elementwise {
                primitive,
                inputs,
                output,
            } => self.lower_elementwise(id, primitive, inputs, output),
            SemanticNodeView::Reduce {
                op,
                axis,
                input,
                output,
                ..
            } => self.lower_reduce(op, axis, input, output),
            SemanticNodeView::Call { .. }
            | SemanticNodeView::If { .. }
            | SemanticNodeView::Loop { .. } => {
                unreachable!("source control is resumed by its owning construction frame")
            }
            SemanticNodeView::Alloc { extents, output } => {
                let axes=extents.iter().map(|value|self.scalar_ref_nat(&ScalarRef::Value(*value))).collect();
                let view = self.builder.portable_allocate_tensor_axes(output,axes);
                let bound = self.stored_allocation(output, view, InitializationState::empty());
                self.values.bind(self.builder.bindings_mut(), output, bound);
            }
            SemanticNodeView::Fill { value, like, output } => {
                let axes=self.bound(like).tensor().extents().to_vec();
                let view = self.builder.portable_allocate_tensor_axes(output,axes);
                self.builder.schedule().fill_constant_any(view, value);
                let bound = self.stored_allocation(output, view, InitializationState::full());
                self.values.bind(self.builder.bindings_mut(), output, bound);
            }
            SemanticNodeView::Copy { input, output } => self.lower_copy(input, output),
            SemanticNodeView::RepresentationConvert { .. } => self.lower_representation_convert(id),
            SemanticNodeView::View {
                base,
                transform,
                extents,
                output,
            } => self.lower_view(base, extents, transform, output),
            SemanticNodeView::ElementRead {
                place,
                indices,
                output,
            } => self.lower_element_read(place, indices, output),
            SemanticNodeView::ElementWrite {
                place,
                indices,
                value,
                output,
            } => self.lower_write_like(place, indices, value, output, None),
            SemanticNodeView::Store {
                destination,
                value,
                output,
            } => self.lower_store(destination, value, output),
            SemanticNodeView::Atomic {
                op,
                place,
                arguments,
                output,
            } => {
                let Some((&value, indices)) = arguments.split_last() else {
                    panic!("checked atomic has no operand")
                };
                self.lower_write_like(place, indices, value, output, Some(op));
            }
            SemanticNodeView::Check { condition, reason } => {
                self.lower_check(id, condition, reason, node.span())
            }
            SemanticNodeView::TuplePack { inputs, output } => {
                let items = inputs.iter().map(|value| self.bound(*value)).collect();
                self.values
                    .bind(self.builder.bindings_mut(), output, Bound::Tuple(items));
            }
            SemanticNodeView::TupleGet {
                tuple,
                index,
                output,
            } => {
                let Bound::Tuple(items) = self.bound(tuple) else {
                    panic!("tuple projection input is not a tuple")
                };
                self.values.bind(
                    self.builder.bindings_mut(),
                    output,
                    items[usize::try_from(index).expect("tuple index does not fit usize")].clone(),
                );
            }
            SemanticNodeView::Extent {
                tensor,
                axis,
                output,
            } => self.lower_extent(tensor, axis, output),
        }
    }

    fn stored_allocation(
        &mut self, value: SemanticValueId, view: AnyBufferView, initial: InitializationState,
    ) -> Bound {
        let SemanticType::Tensor(tensor) = &self.function.value(value).ty else {
            panic!("stored allocation has a non-tensor semantic value")
        };
        let layout=self.builder.portable_layout(view);
        let root=layout.base;
        let axes = layout.extents.iter().map(|axis| self.builder.arena().int_from_nat(*axis)).collect::<Vec<_>>();
        let mut context = InitializationContext::new(self.builder.arena());
        Bound::stored(self.values.contents.root(&mut context, root, StoredView::new(view, layout.extents.clone()), &axes, initial))
    }

    fn bound(&self, value: SemanticValueId) -> Bound {
        self.values.get(self.builder.bindings(), value)
    }
    fn prepared(&mut self, bound: &Bound) -> PreparedArg {
        prepare_scalar(self.builder.arena(), bound.scalar())
    }

    fn kernel_arg(kernel: &mut PortableBuilder<'_, B>, arg: PreparedArg) -> PortableValue {
        match arg {
            PreparedArg::Index(value) => kernel.nat_arg(value),
            PreparedArg::Integer(_) => panic!("unbounded Integer cannot enter a fixed native kernel argument"),
            PreparedArg::Scalar(symbol, dtype) => kernel.scalar_arg(symbol, dtype),
        }
    }

    fn output_target(&mut self, value: SemanticValueId) -> Bound {
        match self.function.value(value).ty.clone() {
            SemanticType::Tensor(_) => unreachable!("tensor destinations are constructed from actual operand geometry"),
            SemanticType::Scalar(_) => {
                let ScalarPublication::Scalar(slot) = self.builder.portable_publish(value) else {
                    panic!("scalar has range publication")
                };
                Bound::Scalar(ScalarBinding::Published(slot))
            }
            SemanticType::Integer | SemanticType::Index { .. } => {
                let ScalarPublication::Quantity(slot) = self.builder.portable_publish(value) else {
                    panic!("quantity has non-quantity publication")
                };
                Bound::Scalar(ScalarBinding::Quantity(slot))
            }
            SemanticType::Range { .. } => {
                let ScalarPublication::Range { start, end } = self.builder.portable_publish(value)
                else {
                    panic!("range has scalar publication")
                };
                Bound::Range {
                    start: Box::new(Bound::Scalar(ScalarBinding::Quantity(start))),
                    end: Box::new(Bound::Scalar(ScalarBinding::Quantity(end))),
                }
            }
            SemanticType::Tuple(_) => panic!("tuple survived semantic leaf normalization"),
            SemanticType::Opaque { .. } => panic!("opaque value escaped a checked portable body"),
            SemanticType::Void => Bound::Unit,
        }
    }

    fn assign(&mut self, source: &Bound, target: &Bound) {
        match (source, target) {
            (Bound::Tensor(source), Bound::Tensor(target)) => {
                if source.stored() != target.stored() {
                    self.copy_tensor(source.stored(), target.stored());
                    let (TensorRealization::Stored(source), TensorRealization::Stored(target)) = (source, target) else { unreachable!("assignment was physically realized") };
                    self.values.contents.copy_allocation(&mut InitializationContext::new(self.builder.arena()), source, target);
                }
            }
            (Bound::Unit, Bound::Unit) => {}
            (Bound::Scalar(_), Bound::Scalar(ScalarBinding::Quantity(slot))) => {
                let value = match prepare_scalar(self.builder.arena(), source.scalar()) {
                    PreparedArg::Integer(value) => HostValueExpr::Integer(value),
                    PreparedArg::Index(value) => HostValueExpr::Natural(value),
                    _ => panic!("quantity assignment has no exact source value"),
                };
                self.builder.schedule().evaluate_host(HostEvaluation {
                    value,
                    to: HostValueDestination::Quantity(*slot),
                    failure: None,
                });
            }
            (Bound::Scalar(_), Bound::Scalar(ScalarBinding::Published(slot))) => {
                let arg = self.prepared(source);
                let mut kernel = self.builder.portable_kernel();
                let value = Self::kernel_arg(&mut kernel, arg);
                let destination = kernel.result_slot(*slot);
                kernel.store_slot(destination, value);
                let kernel = kernel.close();
                self.builder.schedule().launch_sequential(kernel);
            }
            (Bound::Range { start: a, end: b }, Bound::Range { start: x, end: y }) => {
                self.assign(a, x);
                self.assign(b, y);
            }
            (Bound::Tuple(a), Bound::Tuple(b)) => {
                assert_eq!(a.len(), b.len());
                for (a, b) in a.iter().zip(b) {
                    self.assign(a, b);
                }
            }
            _ => panic!("portable assignment categories differ"),
        }
    }

    /// Copies one logical tensor. A raw schedule copy is legal only for two
    /// canonical contiguous views; transformed views are traversed by logical
    /// indices so their strides remain part of the semantics rather than an
    /// executor-side special case.
    fn copy_tensor(&mut self, source: StoredView, destination: StoredView) {
        let source_layout = self.builder.portable_layout(*source.backing());
        let destination_layout = self.builder.portable_layout(*destination.backing());
        if source.steps().is_empty() && destination.steps().is_empty() && source_layout.contiguous && destination_layout.contiguous {
            self.builder.schedule().copy_any(*source.backing(), *destination.backing());
            return;
        }
        assert_eq!(
            source.backing().representation(),
            destination.backing().representation(),
            "checked tensor copy changed representation"
        );
        assert!(
            matches!(
                registry::representation_info(source.backing().representation()).kind,
                RepresentationKind::Dense(_)
            ),
            "a transformed decode-only representation cannot be a copy destination"
        );
        let axes = destination.extents().to_vec();
        let parallel_domain = if self.mode != SemanticMode::AuthoredBackend {
            let extent = self.builder.arena().nat_product(&axes);
            Some(self.independent_domain(extent, "portable tensor copy workgroup size"))
        } else {
            None
        };
        let mut kernel = self.builder.portable_kernel();
        let source = stored_view_in_kernel(&mut kernel, &source, false);
        let destination = stored_view_in_kernel(&mut kernel, &destination, true);
        let mut logical_base = None;
        if let Some(domain) = parallel_domain {
            let (linear, base) = logical_global_id(&mut kernel, true, domain.parallel_extent);
            logical_base = base;
            let extent = kernel.nat_arg(domain.parallel_extent);
            let active = kernel.cmp(CmpOp::Lt, linear, extent);
            let axis_values = axes
                .iter()
                .map(|axis| kernel.nat_arg(*axis))
                .collect::<Vec<_>>();
            kernel.branch(
                active,
                |kernel| {
                    let index = unravel_index(kernel, linear, &axis_values);
                    let value = kernel.tensor_read(&source, &index);
                    kernel.tensor_write(&destination, &index, value);
                    Vec::new()
                },
                |_| Vec::new(),
            );
        } else {
            nested(
                &mut kernel,
                &axes,
                0,
                &mut Vec::new(),
                &mut |kernel, index| {
                    let value = kernel.tensor_read(&source, index);
                    kernel.tensor_write(&destination, index, value);
                },
            );
        }
        let kernel = kernel.close();
        if let Some(domain) = parallel_domain {
            self.launch_semantic(kernel, domain, logical_base);
        } else {
            self.builder.schedule().launch_sequential(kernel);
        }
    }

    fn emit_host_scalar(
        &mut self,
        node: NodeId,
        primitive: &PrimitiveId,
        inputs: &[SemanticValueId],
        out: SemanticValueId,
    ) {
        use seismic_lang::syntax::ast::BinaryOp as B;
        // A literal quantity is already an immutable exact source value. Keep
        // that value as the binding itself, so geometry defined by it cannot
        // acquire an unrelated schedule slot and an artificial entry premise.
        // Public results still use their declared publication destination.
        if matches!(self.function.value(out).ty, SemanticType::Integer)
            && !self.function.results().contains(&out)
        {
            let literal = match primitive {
                PrimitiveId::Symbolic(expression)
                    if matches!(self.builder.arena().view((*expression).into()), NodeView::IntConst(_)) => Some(*expression),
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
                let condition = condition_expr(arena, &operands[0]);
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
                    _ => panic!("checked exact-to-word operation lacks its typed conversion"),
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
        self.builder.schedule().evaluate_host(HostEvaluation { value, to, failure });
        self.values.bind(self.builder.bindings_mut(), out, target);
    }

    fn emit_scalar(
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

    fn lower_primitive(
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
                let invocation_evaluable = self
                    .builder
                    .arena()
                    .free_symbols(AnyExpr::Nat(direct))
                    .iter()
                    .all(|symbol| {
                        matches!(
                            self.builder.arena().symbol_kind(*symbol),
                            SymbolKind::CallDimension(_)
                                | SymbolKind::CallScalar(_)
                                | SymbolKind::TargetConstant(_)
                                | SymbolKind::Decision(_)
                        )
                    });
                if invocation_evaluable {
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
                panic!("structural primitive survived semantic canonicalization: {primitive:?}")
            }
            PrimitiveId::Decode => {
                panic!("decode is tensor-elementwise after semantic canonicalization")
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

    fn lower_intrinsic(
        &mut self,
        intrinsic: IntrinsicId,
        inputs: &[SemanticValueId],
        output: SemanticValueId,
    ) {
        assert_eq!(
            self.mode,
            SemanticMode::AuthoredBackend,
            "portable reference body contains a backend intrinsic"
        );
        let signature = registry::intrinsic_signature(intrinsic);
        assert_eq!(signature.arguments.len(), inputs.len());
        let registry::IntrinsicExecution::WholeTensor { result: 0 } = signature.execution else {
            panic!("enclosing-parallel intrinsic escaped authored segment formation")
        };

        let axes = match &signature.result {
            registry::IntrinsicResultType::Owned { axes, .. } => axes.iter().map(|projection| {
                self.bound(inputs[projection.argument as usize]).tensor().extents()[projection.axis as usize]
            }).collect::<Vec<_>>(),
            _ => panic!("whole-tensor intrinsic requires an owned tensor result"),
        };
        let scalar_target = match signature.result {
            registry::IntrinsicResultType::Scalar(_) => Some(self.output_target(output)),
            _ => None,
        };
        let owned_target = match signature.result {
            registry::IntrinsicResultType::Owned { .. } => {
                Some(self.builder.portable_allocate_tensor_axes(output, axes.clone()))
            }
            _ => None,
        };
        let mut prepared = Vec::with_capacity(inputs.len());
        for (value, argument) in inputs.iter().zip(&signature.arguments) {
            let bound = self.bound(*value);
            prepared.push(match argument.category {
                registry::OperandCategory::Scalar(dtype) => {
                    IntrinsicPrepared::Scalar(self.prepared(&bound), dtype, false)
                }
                registry::OperandCategory::Constant(dtype) => {
                    IntrinsicPrepared::Scalar(self.prepared(&bound), dtype, true)
                }
                registry::OperandCategory::Readable {
                    representation,
                    rank,
                } => IntrinsicPrepared::Place(bound.tensor(), representation, rank, false),
                registry::OperandCategory::Writable {
                    representation,
                    rank,
                } => IntrinsicPrepared::Place(bound.tensor(), representation, rank, true),
                registry::OperandCategory::Opaque { .. } => {
                    panic!("opaque intrinsic operands are not admitted by the active registry")
                }
            });
        }
        let destination_rank = owned_target.map(|view| {
            u32::try_from(self.builder.portable_layout(view).extents.len())
                .expect("semantic tensor rank exceeds u32::MAX")
        });
        let one = self.builder.arena().nat(1);
        let zero = self.builder.arena().nat(0);
        let parallel_extent = self.builder.arena().nat_product(&axes);
        let target = self.builder.portable_target_ref();
        let implementation = self
            .builder
            .portable_registry_ref()
            .intrinsic(signature.id)
            .expect("checked intrinsic is absent from the compiler registry");
        let requirements = (implementation.launch_requirements)(
            target,
            self.builder.arena(),
            signature,
            parallel_extent,
        );
        let workgroup = requirements.required_workgroup.unwrap_or([one, one, one]);
        let participants = self.builder.arena().nat_product(&workgroup);
        let groups = self
            .builder
            .arena()
            .nat_ceil_div(parallel_extent, participants);
        let empty =
            self.builder
                .arena()
                .nat_cmp(seismic_lang::expr::CmpOp::Eq, parallel_extent, zero);
        let domain = SegmentLaunchDomain {
            mode: requirements
                .required_mode
                .unwrap_or(LaunchParticipation::Independent),
            grid: [groups, one, one],
            workgroup,
            empty,
            parallel_extent,
        };
        let mut kernel = self.builder.portable_kernel();
        let mut operands = Vec::with_capacity(prepared.len());
        for prepared in prepared {
            let operand = match prepared {
                IntrinsicPrepared::Scalar(argument, dtype, constant) => {
                    let value = Self::kernel_arg(&mut kernel, argument);
                    let scalar = kernel.semantic_scalar(value, dtype);
                    if constant {
                        SemanticIntrinsicOperand::Constant(scalar)
                    } else {
                        SemanticIntrinsicOperand::Scalar(scalar)
                    }
                }
                IntrinsicPrepared::Place(view, representation, rank, writable) => {
                    assert_eq!(view.backing().representation(), representation);
                    let tensor = stored_view_in_kernel(&mut kernel, &view, writable);
                    let place = kernel.semantic_place(tensor, representation, rank, writable);
                    if writable {
                        SemanticIntrinsicOperand::Writable(place)
                    } else {
                        SemanticIntrinsicOperand::Readable(place)
                    }
                }
            };
            operands.push(operand);
        }
        let destination = owned_target.map(|view| {
            let place = kernel.arg_view(view, true);
            {
                let tensor = kernel.tensor(place);
                kernel.semantic_place(
                    tensor,
                    view.representation(),
                    destination_rank.expect("owned intrinsic has no destination rank"),
                    true,
                )
            }
        });
        let call = SemanticIntrinsicCall {
            signature,
            operands: &operands,
            destination,
        };
        let mut sink = SemanticIntrinsicSink::open(&mut kernel, &call);
        (implementation.lower)(target, &domain, call, &mut sink);
        let result = sink.finish();
        match (result, scalar_target.as_ref()) {
            (
                SemanticIntrinsicResult::Scalar(value),
                Some(Bound::Scalar(ScalarBinding::Published(slot))),
            ) => {
                let destination = kernel.result_slot(*slot);
                kernel.store_slot(destination, value.value());
            }
            (SemanticIntrinsicResult::Owned(_), None) | (SemanticIntrinsicResult::Void, None) => {}
            (SemanticIntrinsicResult::Opaque(_), _) => {
                panic!("opaque intrinsic results are not admitted by the active registry")
            }
            _ => panic!("intrinsic result differs from its checked registry signature"),
        }
        let kernel = kernel.close();
        self.launch_semantic(kernel, domain, None);
        match signature.result {
            registry::IntrinsicResultType::Scalar(_) => {
                self.values.bind(
                    self.builder.bindings_mut(),
                    output,
                    scalar_target.expect("scalar intrinsic has a target"),
                );
            }
            registry::IntrinsicResultType::Owned { .. } => {
                let bound = self.stored_allocation(output, owned_target.expect("owned intrinsic has a target"), InitializationState::full());
                self.values.bind(self.builder.bindings_mut(), output, bound);
            }
            registry::IntrinsicResultType::Void => {
                self.values
                    .bind(self.builder.bindings_mut(), output, Bound::Unit);
            }
            registry::IntrinsicResultType::Opaque { .. } => {
                panic!("opaque intrinsic results are not admitted by the active registry")
            }
        }
    }

    fn initialization_scalar(&mut self, bound: &Bound) -> InitializationArgument {
        match bound {
            Bound::Scalar(scalar) => match prepare_scalar(self.builder.arena(), *scalar) {
                PreparedArg::Index(value) => InitializationArgument::Integer(self.builder.arena().int_from_nat(value)),
                PreparedArg::Integer(value) => InitializationArgument::Integer(value),
                PreparedArg::Scalar(symbol, DType::I32) => {
                    let value = self.builder.arena().scalar_symbol::<seismic_lang::expr::I32>(symbol);
                    InitializationArgument::Integer(self.builder.arena().int_from_scalar(value))
                }
                PreparedArg::Scalar(symbol, DType::U32) => {
                    let value = self.builder.arena().scalar_symbol::<seismic_lang::expr::U32>(symbol);
                    InitializationArgument::Integer(self.builder.arena().int_from_scalar(value))
                }
                PreparedArg::Scalar(_, DType::Bool) => InitializationArgument::Predicate { value: condition_expr(self.builder.arena(), bound), binders: self.values.binders.clone() },
                _ => InitializationArgument::Unknown,
            },
            Bound::Range { start, end } => {
                match (self.initialization_scalar(start), self.initialization_scalar(end)) {
                    (InitializationArgument::Integer(start), InitializationArgument::Integer(end)) => InitializationArgument::Range { start, end },
                    _ => InitializationArgument::Unknown,
                }
            }
            _ => InitializationArgument::Unknown,
        }
    }

    fn initialization_arguments<'v>(&mut self, bounds: &'v [Bound]) -> Vec<CallArgument<'v>> {
        bounds.iter().map(|bound| match bound {
            Bound::Tensor(TensorRealization::Stored(stored)) => CallArgument::Tensor(stored),
            Bound::Tensor(TensorRealization::Computed(plan)) => {
                let axes = plan.axes.iter().map(|axis| self.builder.arena().int_from_nat(*axis)).collect::<Vec<_>>();
                CallArgument::Computed(InitializationContext::new(self.builder.arena()).root(&axes))
            }
            other => CallArgument::Scalar(self.initialization_scalar(other)),
        }).collect()
    }

    fn prepare_call(&mut self, id: NodeId, inputs: &[SemanticValueId]) -> (crate::implementation::CallConstruction<B>, Vec<Bound>) {
        let arguments = inputs.iter().map(|value| self.builder.bindings().selected(self.values.handle(*value), &self.values.selections)).collect::<Vec<_>>();
        let bounds = inputs.iter().map(|value| self.bound(*value)).collect::<Vec<_>>();
        let initialized_arguments = self.initialization_arguments(&bounds);
        let progress = self.builder.begin_call(id, &arguments, &self.values.selections, &self.values.contents, &initialized_arguments, &self.values.binders);
        (progress, bounds)
    }

    fn finish_call(&mut self, progress: crate::implementation::CallConstruction<B>, bounds: &[Bound]) {
        let initialized_arguments = self.initialization_arguments(bounds);
        let call = self.builder.finish_call(progress, &self.values.selections, &mut self.values.contents, &initialized_arguments, &self.values.binders);
        for (output, binding) in &call {
            let slot = self.values.slot(*output);
            assert!(self.values.values[slot].replace(*binding).is_none(), "call result was bound twice");
        }
    }

    fn lower_copy(&mut self, input: SemanticValueId, output: SemanticValueId) {
        let source = self.bound(input).tensor();
        let destination = self.builder.portable_allocate_tensor_axes(output, source.extents().to_vec());
        self.copy_tensor(source, StoredView::new(destination, self.builder.portable_layout(destination).extents));
        let bound = self.stored_allocation(output, destination, InitializationState::full());
        self.values.bind(self.builder.bindings_mut(), output, bound);
    }

    fn lower_representation_convert(&mut self, node: NodeId) {
        let SemanticNodeView::RepresentationConvert {
            conversion,
            input,
            output,
        } = self.function.node(node).view()
        else {
            unreachable!("representation conversion lowering received another node")
        };
        let source = self.bound(input).tensor();
        let destination_view = self.builder.portable_allocate_tensor_axes(output, source.extents().to_vec());
        let recipe = registry::representation_conversion_info(conversion);
        assert_eq!(source.backing().representation(), recipe.source);
        assert_eq!(destination_view.representation(), recipe.destination);
        let RepresentationKind::Packed(layout) =
            &registry::representation_info(recipe.destination).kind
        else {
            panic!(
                "registered representation conversion destination is not resident packed storage"
            )
        };
        let mut packet_axes = source.extents().to_vec();
        let last = packet_axes
            .last_mut()
            .expect("checked packed representation has no packing axis");
        let group = self.builder.arena().nat(u64::from(layout.group));
        *last = self.builder.arena().nat_ceil_div(*last, group);
        let packet_count = self.builder.arena().nat_product(&packet_axes);
        let mut kernel = self.builder.portable_kernel();
        let source = stored_view_in_kernel(&mut kernel, &source, false);
        let destination_place = kernel.representation_destination(destination_view);
        let destination = kernel.tensor(destination_place);
        let packet_count_arg = kernel.nat_arg(packet_count);
        let compiler_owned_base = self.mode != SemanticMode::AuthoredBackend;
        let (packet, logical_base) =
            logical_global_id(&mut kernel, compiler_owned_base, packet_count);
        let active = kernel.cmp(CmpOp::Lt, packet, packet_count_arg);
        kernel.branch(
            active,
            |kernel| {
                kernel.representation_convert_packet(&source, &destination, conversion, packet);
                Vec::new()
            },
            |_| Vec::new(),
        );
        let kernel = kernel.close();
        let domain = if self.mode == SemanticMode::Portable {
            // Representation conversion is independently defined per packet. Keeping
            // it inside one sequential kernel would hide shape-dependent service
            // demand from universal launch specialization, leaving no finite launch
            // boundary for otherwise valid tensors. The universal portable form uses
            // one participant per packet and a compiler-owned logical base, so the
            // ordinary exact inversion/chunking path owns that boundary explicitly.
            let one = self.builder.arena().nat(1);
            let zero = self.builder.arena().nat(0);
            let empty =
                self.builder
                    .arena()
                    .nat_cmp(seismic_lang::expr::CmpOp::Eq, packet_count, zero);
            SegmentLaunchDomain {
                mode: LaunchParticipation::Independent,
                grid: [packet_count, one, one],
                workgroup: [one, one, one],
                empty,
                parallel_extent: packet_count,
            }
        } else {
            self.independent_domain(packet_count, "representation conversion workgroup size")
        };
        self.launch_semantic(kernel, domain, logical_base);
        let bound = self.stored_allocation(output, destination_view, InitializationState::full());
        self.values.bind(self.builder.bindings_mut(), output, bound);
    }

    fn lower_view(
        &mut self,
        base_value: SemanticValueId,
        extents: &[SemanticValueId],
        transform: &ViewTransform,
        out: SemanticValueId,
    ) {
        let Bound::Tensor(TensorRealization::Stored(stored)) = self.bound(base_value) else {
            panic!("physical view needs its stored base")
        };
        let base = stored.view.clone();
        let view = match transform {
            ViewTransform::Identity => base,
            ViewTransform::Transpose { permutation } => base.transpose(permutation.clone()),
            ViewTransform::Reshape { .. } => base.reshape(extents.iter().map(|value|self.scalar_ref_nat(&ScalarRef::Value(*value))).collect()),
            ViewTransform::Slice { axes } => self.lower_slice_view(base, axes),
            ViewTransform::Plane { .. } => panic!("physical plane escaped into portable source"),
        };
        let initialized_view = match transform {
            ViewTransform::Identity => stored.initialized_view.clone(),
            ViewTransform::Transpose { permutation } => InitializationContext::new(self.builder.arena()).permute(&stored.initialized_view, permutation),
            ViewTransform::Reshape { .. } => {
                let axes = view.extents().iter().map(|axis| self.builder.arena().int_from_nat(*axis)).collect::<Vec<_>>();
                InitializationContext::new(self.builder.arena()).reshape(&stored.initialized_view, &axes)
            }
            ViewTransform::Slice { axes } => {
                let mut selections = Vec::with_capacity(axes.len());
                for axis in axes {
                    selections.push(match axis {
                        SliceAxis::Full => (None, None, false),
                        SliceAxis::Point { value, .. } => {
                            let value = self.scalar_ref_nat(value);
                            (Some(self.builder.arena().int_from_nat(value)), None, true)
                        }
                        SliceAxis::Range { start, end, .. } => {
                            let start = start.as_ref().map(|value| { let value = self.scalar_ref_nat(value); self.builder.arena().int_from_nat(value) });
                            let end = end.as_ref().map(|value| { let value = self.scalar_ref_nat(value); self.builder.arena().int_from_nat(value) });
                            (start, end, false)
                        }
                    });
                }
                InitializationContext::new(self.builder.arena()).slice(&stored.initialized_view, &selections)
            }
            ViewTransform::Plane { .. } => unreachable!(),
        };
        let alias = self.values.contents.alias(&stored, view, initialized_view);
        self.values.bind(self.builder.bindings_mut(), out, Bound::stored(alias));
    }

    fn lower_slice_view(&mut self, base: StoredView, axes: &[SliceAxis]) -> StoredView {
        use seismic_ir::tensor_view::SliceAxis as Selection;
        let selections = axes.iter().enumerate().map(|(axis, selection)| match selection {
            SliceAxis::Full => Selection::Full,
            SliceAxis::Point { value, .. } => Selection::Point(self.scalar_ref_nat(value)),
            SliceAxis::Range { start, end, .. } => Selection::Range {
                start: start.as_ref().map(|value| self.scalar_ref_nat(value)).unwrap_or_else(|| self.builder.arena().nat(0)),
                end: end.as_ref().map(|value| self.scalar_ref_nat(value)).unwrap_or(base.extents()[axis]),
            },
        }).collect();
        base.slice(selections, |end, start| self.builder.arena().nat_sub(end, start))
    }

    fn scalar_ref_nat(&mut self, value: &ScalarRef) -> NatExpr {
        match value {
            ScalarRef::Static(value) => *value,
            ScalarRef::Value(value) => match self.prepared(&self.bound(*value)) {
                PreparedArg::Index(value) => value,
                PreparedArg::Integer(value) => self.builder.arena().nat_from_int(value),
                PreparedArg::Scalar(symbol,dtype) => {
                    // The checked slice continuation establishes nonnegative,
                    // in-range coordinates. Preserve the actual source word;
                    // never substitute its producer's unbounded algebra.
                    let integer=match dtype {
                        DType::I32=> {
                            let word=self.builder.arena().scalar_symbol::<seismic_lang::expr::I32>(symbol);
                            self.builder.arena().int_from_scalar(word)
                        }
                        DType::U32=> {
                            let word=self.builder.arena().scalar_symbol::<seismic_lang::expr::U32>(symbol);
                            self.builder.arena().int_from_scalar(word)
                        }
                        _=>panic!("checked slice bound is not an integer word"),
                    };
                    self.builder.arena().nat_from_int(integer)
                },
            },
        }
    }

    fn lower_element_read(
        &mut self,
        place: SemanticValueId,
        indices: &[SemanticValueId],
        output: SemanticValueId,
    ) {
        let view = self.bound(place).tensor();
        let indices = indices
            .iter()
            .map(|value| PreparedArg::Index(self.scalar_ref_nat(&ScalarRef::Value(*value))))
            .collect::<Vec<_>>();
        let target = self.output_target(output);
        let Bound::Scalar(ScalarBinding::Published(slot)) = target else {
            panic!("element read output is not scalar")
        };
        let mut kernel = self.builder.portable_kernel();
        let place = stored_view_in_kernel(&mut kernel, &view, false);
        let indices = indices
            .into_iter()
            .map(|arg| {
                let value = Self::kernel_arg(&mut kernel, arg);
                portable_index(&mut kernel, value)
            })
            .collect::<Vec<_>>();
        let value = kernel.tensor_read(&place, &indices);
        let destination = kernel.result_slot(slot);
        kernel.store_slot(destination, value);
        let kernel = kernel.close();
        self.builder.schedule().launch_sequential(kernel);
        self.values
            .bind(self.builder.bindings_mut(), output, target);
    }
    fn lower_write_like(
        &mut self,
        place: SemanticValueId,
        indices: &[SemanticValueId],
        value: SemanticValueId,
        output: SemanticValueId,
        atomic: Option<seismic_lang::intrinsics::AtomicOp>,
    ) {
        let Bound::Tensor(TensorRealization::Stored(stored)) = self.bound(place) else { panic!("write needs stored place") };
        let base = stored.view.clone();
        let selections = indices.iter().map(|value| {
            let bound = self.bound(*value);
            let InitializationArgument::Integer(index) = self.initialization_scalar(&bound) else {
                panic!("checked element index has no integer value")
            };
            (Some(index), None, true)
        }).collect::<Vec<_>>();
        let initialized_view = InitializationContext::new(self.builder.arena()).slice(&stored.initialized_view, &selections);
        let point = self.values.contents.alias(&stored, base.clone(), initialized_view);
        self.values.contents.write(&mut InitializationContext::new(self.builder.arena()), &point);
        let mut args = indices
            .iter()
            .map(|value| PreparedArg::Index(self.scalar_ref_nat(&ScalarRef::Value(*value))))
            .collect::<Vec<_>>();
        args.push(self.prepared(&self.bound(value)));
        let mut kernel = self.builder.portable_kernel();
        let place = stored_view_in_kernel(&mut kernel, &base, true);
        let value = Self::kernel_arg(&mut kernel, args.pop().expect("write has no value"));
        let destination_type = value_type(element_dtype(base.backing().representation()));
        let value = if value.ty() != destination_type {
            // Checked element assignment permits the language's explicit
            // float-to-float rounding relation. Realize that relation at the
            // storage boundary so dense writes remain exactly typed.
            kernel.cast(value, destination_type)
        } else {
            value
        };
        let indices = args
            .into_iter()
            .map(|arg| {
                let value = Self::kernel_arg(&mut kernel, arg);
                portable_index(&mut kernel, value)
            })
            .collect::<Vec<_>>();
        match atomic {
            Some(op) => kernel.tensor_atomic(op, &place, &indices, value),
            None => kernel.tensor_write(&place, &indices, value),
        }
        let kernel = kernel.close();
        self.builder.schedule().launch_sequential(kernel);
        self.values
            .bind(self.builder.bindings_mut(), output, Bound::stored(stored));
    }

    fn lower_extent(&mut self, tensor: SemanticValueId, axis: u32, output: SemanticValueId) {
        let view = self.bound(tensor).tensor();
        let target = self.output_target(output);
        let value = *view.extents().get(axis as usize)
            .expect("checked extent axis exceeds actual view rank");
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

    fn lower_check(
        &mut self,
        node: NodeId,
        condition_value: SemanticValueId,
        reason: &CheckReason,
        span: seismic_lang::span::Span,
    ) {
        let arg = self.prepared(&self.bound(condition_value));
        let slot = self.builder.schedule().temporary_bool();
        let mut kernel = self.builder.portable_kernel();
        let condition = Self::kernel_arg(&mut kernel, arg);
        let destination = kernel.result_slot(slot);
        kernel.store_slot(destination, condition);
        let kernel = kernel.close();
        self.builder.schedule().launch_sequential(kernel);
        self.builder.schedule().check_any(
            slot,
            CheckSite {
                failure: SourceFailure::at(self.function, node, SourceFailureCause::Check(reason.clone())),
                path: self.function.name().to_owned(),
                line: span.start,
            },
        );
    }

    fn lower_store(
        &mut self,
        destination_value: SemanticValueId,
        value: SemanticValueId,
        output: SemanticValueId,
    ) {
        let Bound::Tensor(TensorRealization::Stored(stored)) = self.bound(destination_value) else { panic!("store needs stored destination") };
        let destination = stored.view.clone();
        let mut source = self.bound(value).tensor();
        if self.builder.portable_views_may_overlap(*source.backing(),*destination.backing()) {
            // Store evaluates its complete RHS before writing the selected place.
            // A selected descriptor may alias even when its view handle differs.
            let axes=source.extents().to_vec();
            let snapshot=self.builder.allocate_tensor_product(&TensorSemantics {
                representation: source.backing().representation(), axes: axes.clone(), storage: TensorStorage::Owned,
            });
            let snapshot=StoredView::new(snapshot,axes);
            self.copy_tensor(source,snapshot.clone());
            source=snapshot;
        }
        self.copy_tensor(source, destination);
        let base = match &self.function.value(destination_value).ty {
            SemanticType::Tensor(TensorSemantics {
                storage: TensorStorage::View { base, .. },
                ..
            }) => self.bound(*base),
            _ => panic!("store destination is not an explicit writable view"),
        };
        self.values.contents.write(&mut InitializationContext::new(self.builder.arena()), &stored);
        self.values.bind(self.builder.bindings_mut(), output, base);
    }

    fn broadcast_axes<'v>(&mut self, inputs: impl Iterator<Item=&'v [NatExpr]>) -> Vec<NatExpr> {
        let inputs = inputs.collect::<Vec<_>>();
        let rank = inputs.iter().map(|axes| axes.len()).max().expect("tensor operation has a tensor operand");
        let one = self.builder.arena().nat(1);
        let mut output = vec![one;rank];
        for input in inputs {
            let skip = rank-input.len();
            for (target,source) in output[skip..].iter_mut().zip(input) {
                let singleton = self.builder.arena().nat_cmp(seismic_lang::expr::CmpOp::Eq,*target,one);
                *target = self.builder.arena().nat_select(singleton,*source,*target);
            }
        }
        output
    }

    fn lower_elementwise(
        &mut self,
        node: NodeId,
        primitive: &PrimitiveId,
        inputs: &[SemanticValueId],
        out: SemanticValueId,
    ) {
        let input_bounds = inputs
            .iter()
            .map(|value| self.bound(*value))
            .collect::<Vec<_>>();
        let scalar_args = input_bounds
            .iter()
            .map(|bound| match bound {
                Bound::Scalar(_) => Some(self.prepared(bound)),
                _ => None,
            })
            .collect::<Vec<_>>();
        let input_axes = input_bounds.iter().map(|bound| match bound {
            Bound::Tensor(value) => Some(value.stored().extents().to_vec()),
            _ => None,
        }).collect::<Vec<_>>();
        let axes = self.broadcast_axes(input_axes.iter().flatten().map(Vec::as_slice));
        let destination = self.builder.portable_allocate_tensor_axes(out, axes.clone());
        let parallel_domain = if self.mode != SemanticMode::AuthoredBackend {
            let extent = self.builder.arena().nat_product(&axes);
            Some(self.independent_domain(extent, "portable elementwise workgroup size"))
        } else {
            None
        };
        let checks=source_scalar::checks(self.function,node,primitive,inputs);
        let statuses=self.builder.portable_source_statuses(checks.len());
        let mut kernel = self.builder.portable_kernel();
        let status_words=segment_check_statuses(&mut kernel,&checks,&statuses);
        let output = kernel.arg_view(destination, true);
        let places = input_bounds
            .iter()
            .map(|bound| match bound {
                Bound::Tensor(value) => Some(stored_view_in_kernel(&mut kernel, &value.stored(), false)),
                _ => None,
            })
            .collect::<Vec<_>>();
        let scalars = scalar_args
            .into_iter()
            .map(|arg| arg.map(|arg| Self::kernel_arg(&mut kernel, arg)))
            .collect::<Vec<_>>();
        let mut emit = |kernel: &mut PortableBuilder<'_, B>, index: &[PortableValue],alive:PortableValue| {
            kernel.branch(alive,|kernel| {
            let mut args = Vec::new();
            for ((place, scalar), source_axes) in
                places.iter().zip(&scalars).zip(&input_axes)
            {
                if let Some(place) = place {
                    let source_axes = source_axes.as_ref().expect("tensor input has no axes");
                    let skip = index.len() - source_axes.len();
                    let source_index = index[skip..]
                        .iter().zip(source_axes).map(|(value, extent)| {
                            let extent = kernel.nat_arg(*extent);
                            let one = kernel.index_constant(1);
                            let zero = kernel.index_constant(0);
                            let singleton = kernel.cmp(CmpOp::Eq,extent,one);
                            kernel.select(singleton,zero,*value)
                        }).collect::<Vec<_>>();
                    args.push(kernel.tensor_read(place, &source_index));
                } else {
                    args.push(scalar.expect("elementwise input is neither tensor nor scalar"));
                }
            }
            let (value,successful) = source_scalar::lower(
                kernel, self.function, node, &status_words, alive, primitive, &args,
                &SemanticType::Scalar(element_dtype(destination.representation())),
            );
            kernel.branch(successful,|kernel| { kernel.write(output,index,value);Vec::new() },|_|Vec::new());
            vec![successful]
            },|_|vec![alive])[0]
        };
        let alive=kernel.constant(ConstantValue::Bool(true),ValueType::Bool);
        let mut logical_base = None;
        if let Some(domain) = parallel_domain {
            let (linear, base) = logical_global_id(&mut kernel, true, domain.parallel_extent);
            logical_base = base;
            let extent = kernel.nat_arg(domain.parallel_extent);
            let active = kernel.cmp(CmpOp::Lt, linear, extent);
            let axis_values = axes
                .iter()
                .map(|axis| kernel.nat_arg(*axis))
                .collect::<Vec<_>>();
            kernel.branch(
                active,
                |kernel| {
                    let index = unravel_index(kernel, linear, &axis_values);
                    emit(kernel, &index,alive);
                    Vec::new()
                },
                |_| Vec::new(),
            );
        } else {
            let axes=axes.iter().map(|axis|kernel.nat_arg(*axis)).collect::<Vec<_>>();
            source_scalar::nested(&mut kernel,&axes,0,&mut Vec::new(),alive,&mut emit);
        }
        let kernel = kernel.close();
        if let Some(domain) = parallel_domain {
            self.launch_semantic(kernel, domain, logical_base);
        } else {
            self.builder.schedule().launch_sequential(kernel);
        }
        source_scalar::finish(self.builder,self.function.name(),checks,statuses);
        let bound = self.stored_allocation(out, destination, InitializationState::full());
        self.values.bind(self.builder.bindings_mut(), out, bound);
    }

    fn stream_tensor_plan(&self, value: SemanticValueId) -> Arc<StreamTensorPlan> {
        let axes = match self.bound(value) {
            Bound::Tensor(TensorRealization::Stored(tensor)) => tensor.view.extents().to_vec(),
            Bound::Tensor(TensorRealization::Computed(tensor)) => tensor.axes.clone(),
            _ => panic!("tensor consumer received a non-tensor value"),
        };
        Arc::new(StreamTensorPlan {
            axes,
            // Capture the actual dominating binding, rather than recursively
            // cloning its current realization. Frozen call products substitute
            // parameter handles through this same edge.
            value: TensorDefinitionValue::Physical(self.values.handle(value)),
        })
    }

    fn computed_tensor(&mut self, value: SemanticValueId) -> Arc<StreamTensorPlan> {
        let SemanticType::Tensor(tensor_type) = &self.function.value(value).ty else {
            unreachable!("computed tensor producer has non-tensor semantics")
        };
        let mut axes = tensor_type.axes.clone();
        let ValueOrigin::Node(node) = self.function.value(value).origin else {
            panic!("streamed tensor has no pure producer")
        };
        let plan = match self.function.node(node).view() {
            SemanticNodeView::Elementwise {
                primitive,
                inputs,
                output,
            } => {
                assert_eq!(output, value, "streamed producer output mismatch");
                let inputs = inputs
                    .iter()
                    .map(|input| match &self.function.value(*input).ty {
                        SemanticType::Tensor(_) => {
                            StreamBoundPlan::Tensor(self.stream_tensor_plan(*input))
                        }
                        SemanticType::Scalar(_) | SemanticType::Index { .. } => {
                            StreamBoundPlan::Scalar(self.prepared(&self.bound(*input)))
                        }
                        _ => panic!("streamed elementwise input is not scalar or tensor"),
                    })
                    .collect::<Vec<_>>();
                axes = self.broadcast_axes(inputs.iter().filter_map(|input| match input {
                    StreamBoundPlan::Tensor(tensor) => Some(tensor.axes.as_slice()),
                    _ => None,
                }));
                let mut tensor = tensor_type.clone();
                tensor.axes = axes.clone();
                let result = self.bind_tensor_result(value, tensor);
                TensorDefinitionValue::Elementwise {
                    primitive: primitive.clone(), inputs, result,
                }
            }
            SemanticNodeView::View {
                base,
                transform,
                extents,
                output,
            } => {
                assert_eq!(output, value, "streamed view output mismatch");
                if matches!(transform,ViewTransform::Reshape { .. }) {
                    axes=extents.iter().map(|value|self.scalar_ref_nat(&ScalarRef::Value(*value))).collect();
                }
                let base = self.stream_tensor_plan(base);
                match transform {
                    ViewTransform::Identity => return base,
                    ViewTransform::Transpose { permutation } => {
                        axes = permutation.iter().map(|axis| base.axes[*axis as usize]).collect();
                        TensorDefinitionValue::View { base, transform: StreamViewPlan::Transpose(permutation.clone()) }
                    },
                    ViewTransform::Reshape { .. } => TensorDefinitionValue::View {
                        base,
                        transform: StreamViewPlan::Reshape,
                    },
                    ViewTransform::Slice { axes: selections } => {
                        let mut actual = Vec::new();
                        for (axis, selection) in selections.iter().enumerate() {
                            match selection {
                                SliceAxis::Full => actual.push(base.axes[axis]),
                                SliceAxis::Point { .. } => (),
                                SliceAxis::Range { start, end, .. } => {
                                    let start = start.as_ref().map(|value| self.scalar_ref_nat(value)).unwrap_or_else(|| self.builder.arena().nat(0));
                                    let end = end.as_ref().map(|value| self.scalar_ref_nat(value)).unwrap_or(base.axes[axis]);
                                    actual.push(self.builder.arena().nat_sub(end,start));
                                }
                            }
                        }
                        axes = actual;
                        let zero = self.builder.arena().nat(0);
                        let axes = selections
                            .iter()
                            .map(|axis| match axis {
                                SliceAxis::Full => StreamSliceAxisPlan::Full,
                                SliceAxis::Point { value, .. } => {
                                    StreamSliceAxisPlan::Point(self.stream_scalar_ref(value))
                                }
                                SliceAxis::Range { start, .. } => StreamSliceAxisPlan::Range {
                                    start: start
                                        .as_ref()
                                        .map(|value| self.stream_scalar_ref(value))
                                        .unwrap_or(PreparedArg::Index(zero)),
                                },
                            })
                            .collect();
                        TensorDefinitionValue::View {
                            base,
                            transform: StreamViewPlan::Slice(axes),
                        }
                    }
                    ViewTransform::Plane { .. } => {
                        panic!("physical plane view cannot be a streamed semantic expression")
                    }
                }
            }
            _ => panic!("streamed tensor producer is not pure elementwise or view"),
        };
        Arc::new(StreamTensorPlan { axes, value: plan })
    }

    fn realize_tensor_at(&mut self, plan: &StreamTensorPlan, destination: AnyBufferView) {
        let extent = self.builder.arena().nat_product(&plan.axes);
        let domain = self.independent_domain(extent, "tensor materialization workgroup size");
        let (mut kernel, bindings) = self.builder.portable_kernel_with_bindings();
        let source =
            instantiate_stream_tensor(&mut kernel, &plan, bindings, &self.values.selections);
        let output = kernel.arg_view(destination, true);
        let (linear, binding) = logical_global_id(&mut kernel, true, extent);
        let end = kernel.nat_arg(extent);
        let active = kernel.cmp(CmpOp::Lt, linear, end);
        kernel.branch(
            active,
            |kernel| {
                let index = unravel_index(kernel, linear, &source.axes);
                let value = source.read(kernel, &index);
                kernel.write(output, &index, value);
                Vec::new()
            },
            |_| Vec::new(),
        );
        let kernel = kernel.close();
        self.launch_semantic(kernel, domain, binding);
    }

    fn materialize_tensor(&mut self, value: SemanticValueId) {
        let Bound::Tensor(TensorRealization::Computed(plan)) = self.values.get(self.builder.bindings(), value) else { return; };
        let destination = self.builder.portable_allocate_tensor_axes(value, plan.axes.clone());
        self.realize_tensor_at(&plan,destination);
        let bound=self.stored_allocation(value,destination,InitializationState::full());
        self.values.rebind(self.builder.bindings_mut(),value,bound);
    }


    /// Realize an owned product at an affine transport boundary. Affine maps
    /// preserve storage; nonaffine maps relocate bytes and initialized regions.
    fn realize_region_product(&mut self, bound: Bound, ty: &SemanticType) -> Bound {
        self.realize_region_product_with(bound,ty,&mut BTreeMap::new())
    }
    fn realize_region_product_with(&mut self, bound: Bound, ty: &SemanticType, realized: &mut BTreeMap<*const StreamTensorPlan,StoredTensor>) -> Bound {
        match (bound,ty) {
            (Bound::Tensor(TensorRealization::Computed(plan)),SemanticType::Tensor(tensor)) => {
                let identity=Arc::as_ptr(&plan);
                if let Some(stored)=realized.get(&identity) { return Bound::stored(stored.clone()); }
                let mut tensor = tensor.clone();
                tensor.axes = plan.axes.clone();
                let destination=self.builder.allocate_tensor_product(&tensor);
                self.realize_tensor_at(&plan,destination);
                let root=self.builder.portable_layout(destination).base;
                let axes=tensor.axes.iter().map(|axis|self.builder.arena().int_from_nat(*axis)).collect::<Vec<_>>();
                let stored=self.values.contents.root(&mut InitializationContext::new(self.builder.arena()),root,StoredView::new(destination,tensor.axes.clone()),&axes,InitializationState::full());
                realized.insert(identity,stored.clone());
                Bound::stored(stored)
            }
            (Bound::Tensor(TensorRealization::Stored(source)),SemanticType::Tensor(tensor)) if !source.view.steps().is_empty() => {
                if let Some(view)=self.builder.portable_affine_view(&source.view) {
                    let mapped=StoredView::new(view,source.view.extents().to_vec());
                    return Bound::stored(self.values.contents.alias(&source,mapped,source.initialized_view.clone()));
                }
                // This owned transport boundary may include unspecified
                // elements. Relocate storage bits, then transfer the exact
                // initialized region through the same logical map.
                let mut tensor=tensor.clone();
                tensor.axes=source.view.extents().to_vec();
                let destination=self.builder.allocate_tensor_product(&tensor);
                let mapped=StoredView::new(destination,tensor.axes.clone());
                self.builder.portable_relocate_tensor(&source.view,destination);
                let root=self.builder.portable_layout(destination).base;
                let axes=tensor.axes.iter().map(|axis|self.builder.arena().int_from_nat(*axis)).collect::<Vec<_>>();
                let stored=self.values.contents.root(&mut InitializationContext::new(self.builder.arena()),root,mapped,&axes,InitializationState::empty());
                self.values.contents.copy_allocation(&mut InitializationContext::new(self.builder.arena()),&source,&stored);
                Bound::stored(stored)
            }
            (Bound::Tuple(values),SemanticType::Tuple(types)) => {
                assert_eq!(values.len(),types.len());
                Bound::Tuple(values.into_iter().zip(types).map(|(value,ty)|self.realize_region_product_with(value,ty,realized)).collect())
            }
            (bound,_)=>bound,
        }
    }

    fn region_operand_product(&mut self, bound: &Bound) -> seismic_ir::region::Product<seismic_ir::region::ValueOperand> {
        use seismic_ir::region::{Product,ValueOperand,ScalarOperand};
        match bound {
            Bound::Unit=>Product::Unit,
            Bound::Tensor(value)=>Product::Leaf(ValueOperand::Tensor(*value.stored().direct_backing().expect("mapped carry requires owned realization before affine region transport"))),
            Bound::Scalar(ScalarBinding::Quantity(slot)) => {
                use seismic_ir::region::QuantityOperand;
                let operand = match slot.kind() {
                    seismic_ir::schedule::HostQuantityKind::Integer => QuantityOperand::Integer(self.builder.arena().int_symbol(slot.symbol())),
                    seismic_ir::schedule::HostQuantityKind::Natural => QuantityOperand::Natural(self.builder.arena().nat_symbol(slot.symbol())),
                };
                Product::Leaf(ValueOperand::Quantity(operand))
            }
            Bound::Scalar(value) => match prepare_scalar(self.builder.arena(), *value) {
                PreparedArg::Index(value) => Product::Leaf(ValueOperand::Quantity(
                    seismic_ir::region::QuantityOperand::Natural(value),
                )),
                PreparedArg::Integer(value) => Product::Leaf(ValueOperand::Quantity(
                    seismic_ir::region::QuantityOperand::Integer(value),
                )),
                PreparedArg::Scalar(symbol, dtype) => Product::Leaf(ValueOperand::Scalar(
                    ScalarOperand::Word { symbol, dtype },
                )),
            },
            Bound::Range{start,end}=>Product::Range(Box::new(self.region_operand_product(start)),Box::new(self.region_operand_product(end))),
            Bound::Tuple(values)=>Product::Tuple(values.iter().map(|value|self.region_operand_product(value)).collect()),
        }
    }

    fn region_destination_product(
        &mut self,
        destination: &seismic_ir::region::Product<seismic_ir::region::ValueDestination>,
        source: &Bound,
        contents: &StorageContents,
        alternate: Option<(&Bound,&StorageContents)>,
    ) -> Bound {
        use seismic_ir::region::{Product,ValueDestination};
        match (destination,source) {
            (Product::Unit,Bound::Unit)=>Bound::Unit,
            (Product::Leaf(ValueDestination::Scalar(slot)),Bound::Scalar(_))=>Bound::Scalar(ScalarBinding::Published(*slot)),
            (Product::Leaf(ValueDestination::Quantity(slot)),Bound::Scalar(_))=>Bound::Scalar(ScalarBinding::Quantity(*slot)),
            (Product::Leaf(ValueDestination::Tensor(view)),Bound::Tensor(TensorRealization::Stored(source)))=> {
                let layout=self.builder.portable_layout(*view).clone();
                let axes=layout.extents.iter().map(|axis|self.builder.arena().int_from_nat(*axis)).collect::<Vec<_>>();
                let mut context=InitializationContext::new(self.builder.arena());
                let mut state=context.project(contents.state(source),&source.initialized_view);
                if let Some((Bound::Tensor(TensorRealization::Stored(other)),other_contents))=alternate {
                    state=state.intersection(&context.project(other_contents.state(other),&other.initialized_view));
                }
                Bound::stored(self.values.contents.root(&mut context,layout.base,StoredView::new(*view,layout.extents.clone()),&axes,state))
            }
            (Product::Range(a,b),Bound::Range{start,end})=> {
                let alternatives=alternate.map(|(other,contents)| {
                    let Bound::Range{start,end}=other else { panic!("range carry changed shape") };
                    ((start.as_ref(),contents),(end.as_ref(),contents))
                });
                Bound::Range{start:Box::new(self.region_destination_product(a,start,contents,alternatives.map(|a|a.0))),end:Box::new(self.region_destination_product(b,end,contents,alternatives.map(|a|a.1)))}
            }
            (Product::Tuple(destinations),Bound::Tuple(values))=> {
                assert_eq!(destinations.len(),values.len());
                let other=alternate.map(|(other,contents)| {
                    let Bound::Tuple(values)=other else { panic!("tuple carry changed shape") };
                    (values,contents)
                });
                Bound::Tuple(destinations.iter().zip(values).enumerate().map(|(index,(destination,value))|self.region_destination_product(destination,value,contents,other.map(|(values,contents)|(&values[index],contents)))).collect())
            }
            _=>panic!("region product transport changed shape"),
        }
    }

    fn stream_scalar_ref(&mut self, value: &ScalarRef) -> PreparedArg {
        match value {
            ScalarRef::Static(value) => PreparedArg::Index(*value),
            ScalarRef::Value(value) => self.prepared(&self.bound(*value)),
        }
    }

    fn lower_reduce(
        &mut self,
        op: ReduceOp,
        axis: u32,
        input_value: SemanticValueId,
        out: SemanticValueId,
    ) {
        let source_plan = self.stream_tensor_plan(input_value);
        let input_axes = source_plan.axes.clone();
        let mut output_axes = input_axes.clone();
        output_axes.remove(axis as usize);
        let destination = if matches!(self.function.value(out).ty,SemanticType::Tensor(_)) {
            let view = self.builder.portable_allocate_tensor_axes(out,output_axes.clone());
            self.stored_allocation(out,view,InitializationState::empty())
        } else {
            assert!(output_axes.is_empty(), "scalar reduction retains no logical axes");
            self.output_target(out)
        };
        let input_dtype = match &self.function.value(input_value).ty {
            SemanticType::Tensor(t) => registry::representation_info(t.representation).decoded,
            _ => panic!("streamed reduction input is not tensor"),
        };
        let schema = reduce_schema(op, input_dtype);
        let extent = self.builder.arena().nat_product(&output_axes);
        let domain = (self.mode != SemanticMode::AuthoredBackend
            && matches!(destination, Bound::Tensor(_)))
        .then(|| self.independent_domain(extent, "portable reduction workgroup size"));
        let (mut kernel, bindings) = self.builder.portable_kernel_with_bindings();
        let source =
            instantiate_stream_tensor(&mut kernel, &source_plan, bindings, &self.values.selections);
        let output = match &destination {
            Bound::Tensor(value) => Some(stored_view_in_kernel(&mut kernel, &value.stored(), true)),
            _ => None,
        };
        let scalar_output = match &destination {
            Bound::Scalar(ScalarBinding::Published(slot)) => Some(kernel.result_slot(*slot)),
            _ => None,
        };
        let emit = |kernel: &mut PortableBuilder<'_, B>, outer: &[PortableValue]| {
            let zero = kernel.index_constant(0);
            let end = kernel.nat_arg(input_axes[axis as usize]);
            let initial = if op == ReduceOp::Sum {
                zero_of(kernel, value_type(schema.accumulator))
            } else {
                let first_index = reduction_index(outer, axis as usize, zero);
                let first = source.read(kernel, &first_index);
                if first.ty() == value_type(schema.accumulator) {
                    first
                } else {
                    kernel.cast(first, value_type(schema.accumulator))
                }
            };
            let start = if matches!(op, ReduceOp::Sum) {
                zero
            } else {
                kernel.index_constant(1)
            };
            let carries = if op == ReduceOp::Argmax {
                vec![initial, kernel.index_constant(0)]
            } else {
                vec![initial]
            };
            let coordinates = uniformity::all(outer.iter().map(|v| kernel.uniformity(*v)));
            let recurrence = vec![source.reduction_uniformity(kernel, coordinates); carries.len()];
            let result =
                kernel.repeat(start, end, carries, &recurrence, |kernel, binder, carry| {
                    let index = reduction_index(outer, axis as usize, binder);
                    let value = source.read(kernel, &index);
                    let value = if value.ty() == value_type(schema.accumulator) {
                        value
                    } else {
                        kernel.cast(value, value_type(schema.accumulator))
                    };
                    match op {
                        ReduceOp::Sum => vec![kernel.binary(BinaryOp::Add, carry[0], value)],
                        ReduceOp::Max => vec![kernel.binary(BinaryOp::Max, carry[0], value)],
                        ReduceOp::Min => vec![kernel.binary(BinaryOp::Min, carry[0], value)],
                        ReduceOp::Argmax => {
                            let better = kernel.cmp(CmpOp::Gt, value, carry[0]);
                            vec![
                                kernel.select(better, value, carry[0]),
                                kernel.select(better, binder, carry[1]),
                            ]
                        }
                    }
                });
            let value = if op == ReduceOp::Argmax {
                kernel.cast(result[1], ValueType::Scalar(DType::I32))
            } else {
                result[0]
            };
            if let Some(output) = &output {
                kernel.tensor_write(output, outer, value);
            } else {
                kernel.store_slot(
                    scalar_output.expect("scalar reduction has a publication slot"),
                    value,
                );
            }
        };
        let logical_base = if let Some(domain) = domain {
            let (linear, base) = logical_global_id(&mut kernel, true, domain.parallel_extent);
            let extent = kernel.nat_arg(domain.parallel_extent);
            let active = kernel.cmp(CmpOp::Lt, linear, extent);
            let axis_values = output_axes
                .iter()
                .map(|axis| kernel.nat_arg(*axis))
                .collect::<Vec<_>>();
            kernel.branch(
                active,
                |kernel| {
                    let outer = unravel_index(kernel, linear, &axis_values);
                    emit(kernel, &outer);
                    Vec::new()
                },
                |_| Vec::new(),
            );
            base
        } else {
            nested(&mut kernel, &output_axes, 0, &mut Vec::new(), &mut { emit });
            None
        };
        let kernel = kernel.close();
        if let Some(domain) = domain {
            self.launch_semantic(kernel, domain, logical_base);
        } else {
            self.builder.schedule().launch_sequential(kernel);
        }
        self.values
            .bind(self.builder.bindings_mut(), out, destination);
    }

    fn begin_if(
        &mut self, condition_value: SemanticValueId, capture_values: &[SemanticValueId],
        outputs: &[SemanticValueId], then_region: RegionId, else_region: RegionId,
    ) -> construction::IfConstruction {
        let condition_bound = self.bound(condition_value);
        let condition = condition_expr(self.builder.arena(), &condition_bound);
        let captures = capture_values.iter().map(|value| self.values.handle(*value)).collect::<Vec<_>>();
        let parent = self.values.clone();
        let schedule = self.builder.begin_source_branch(condition);
        self.values.selections.insert(condition, 1);
        self.bind_if_parameters(then_region, &captures);
        construction::IfConstruction { schedule, condition, captures, outputs: outputs.to_vec(), then_region, else_region, parent, then_products: Vec::new(), then_contents: None }
    }

    fn bind_if_parameters(&mut self, region: RegionId, captures: &[BindingId]) {
        let parameters = self.function.region(region).parameters();
        assert_eq!(parameters.len(), captures.len());
        for (parameter, capture) in parameters.iter().zip(captures) {
            let slot = self.values.slot(*parameter);
            assert!(self.values.values[slot].replace(*capture).is_none());
        }
    }

    fn realized_region_results(&mut self, region: RegionId) -> Vec<Bound> {
        let ids = self.function.region(region).results().to_vec();
        let values = Bound::Tuple(ids.iter().map(|value|self.bound(*value)).collect());
        let types = SemanticType::Tuple(ids.iter().map(|value|self.function.value(*value).ty.clone()).collect());
        let Bound::Tuple(values) = self.realize_region_product(values, &types) else { unreachable!() };
        values
    }

    /// Descriptor forwarding changes physical location only. The selected
    /// source arm keeps its actual storage origin and initialized view.
    fn forward_region_product(&mut self, destination: &seismic_ir::region::Product<seismic_ir::region::ValueDestination>, source: &Bound) -> Bound {
        use seismic_ir::region::{Product,ValueDestination};
        match (destination,source) {
            (Product::Unit,Bound::Unit)=>Bound::Unit,
            (Product::Leaf(ValueDestination::Scalar(slot)),Bound::Scalar(_))=>Bound::Scalar(ScalarBinding::Published(*slot)),
            (Product::Leaf(ValueDestination::Quantity(slot)),Bound::Scalar(_))=>Bound::Scalar(ScalarBinding::Quantity(*slot)),
            (Product::Leaf(ValueDestination::Tensor(view)),Bound::Tensor(TensorRealization::Stored(source)))=>{
                let layout=self.builder.portable_layout(*view);
                let mut value=source.clone();
                value.view=StoredView::new(*view,layout.extents);
                Bound::stored(value)
            }
            (Product::Range(a,b),Bound::Range{start,end})=>Bound::Range {start:Box::new(self.forward_region_product(a,start)),end:Box::new(self.forward_region_product(b,end))},
            (Product::Tuple(a),Bound::Tuple(b))=>{
                assert_eq!(a.len(),b.len());
                Bound::Tuple(a.iter().zip(b).map(|(a,b)|self.forward_region_product(a,b)).collect())
            }
            _=>panic!("branch product changed shape"),
        }
    }

    fn next_if(&mut self, branch: &mut construction::IfConstruction) {
        branch.then_products = self.realized_region_results(branch.then_region);
        branch.then_contents = Some(self.values.contents.clone());
        assert_eq!(self.builder.next_source_branch(&mut branch.schedule), Some(0));
        self.values = branch.parent.clone();
        self.values.selections.insert(branch.condition, 0);
        self.bind_if_parameters(branch.else_region, &branch.captures);
    }

    fn finish_if(&mut self, branch: construction::IfConstruction) {
        let otherwise = self.realized_region_results(branch.else_region);
        let else_contents = self.values.contents.clone();
        let then_values = Bound::Tuple(branch.then_products.clone());
        let else_values = Bound::Tuple(otherwise.clone());
        let then_operands = self.region_operand_product(&then_values);
        let else_operands = self.region_operand_product(&else_values);
        let destination = self.builder.finish_value_branch(branch.schedule, then_operands, else_operands);
        let Bound::Tuple(then_values) = self.forward_region_product(&destination, &then_values) else { unreachable!() };
        let Bound::Tuple(else_values) = self.forward_region_product(&destination, &else_values) else { unreachable!() };
        self.values = branch.parent;
        self.values.contents.branch(&mut InitializationContext::new(self.builder.arena()), branch.condition, &self.values.binders, &branch.then_contents.expect("then arm completed before else arm"), &else_contents);
        assert_eq!(branch.outputs.len(), branch.then_products.len());
        assert_eq!(branch.outputs.len(), otherwise.len());
        for ((output, then), otherwise) in branch.outputs.into_iter().zip(then_values).zip(else_values) {
            let then = self.builder.bindings_mut().insert(then);
            let otherwise = self.builder.bindings_mut().insert(otherwise);
            let selected = self.builder.bindings_mut().selected_value(branch.condition, vec![(1, then), (0, otherwise)]);
            let slot = self.values.slot(output);
            assert!(self.values.values[slot].replace(selected).is_none());
        }
    }

    fn begin_loop(&mut self,start_value:SemanticValueId,end_value:SemanticValueId,capture_values:&[SemanticValueId],body:RegionId,carries:&[seismic_lang::entry::Carry])->LoopConstruction {
        let start_bound=self.bound(start_value);
        let end_bound=self.bound(end_value);
        let start=index_expr(self.builder.arena(),&start_bound);
        let end=index_expr(self.builder.arena(),&end_bound);
        let captures=capture_values.iter().map(|value|self.bound(*value)).collect::<Vec<_>>();
        let initial=Bound::Tuple(carries.iter().map(|carry|self.bound(carry.initial)).collect());
        let initial_type=SemanticType::Tuple(carries.iter().map(|carry|self.function.value(carry.initial).ty.clone()).collect());
        let initial=self.realize_region_product(initial,&initial_type);
        let operands=self.region_operand_product(&initial);
        let parent=self.values.clone();
        let schedule=self.builder.begin_value_repeat(start,end,operands);
        let binding=schedule.binding();
        self.values.binders.push(binding.symbol);
        self.bind_region_parameters(body,&captures);
        let RegionKind::LoopBody{binder_value,..}=self.function.region(body).kind() else {panic!("loop body kind")};
        let index=self.builder.arena().nat_symbol(binding.symbol);
        self.values.bind(self.builder.bindings_mut(),*binder_value,Bound::Scalar(ScalarBinding::Index(index)));
        let Bound::Tuple(headers)=self.region_destination_product(schedule.header(),&initial,&parent.contents,None) else {unreachable!()};
        for (carry,header) in carries.iter().zip(headers) {
            self.values.rebind(self.builder.bindings_mut(),carry.parameter,header);
        }
        let loop_values=self.function.region(body).parameters().iter().map(|value|self.bound(*value)).collect::<Vec<_>>();
        let arguments=self.initialization_arguments(&loop_values);
        let start_integer=self.builder.arena().int_from_nat(start);
        let current_integer=self.builder.arena().int_from_nat(index);
        let mut context=initialization_context(self.builder.arena(),&self.values.selections,&self.values.binders);
        self.values.contents.loop_header(&mut context,self.function.region(body).loop_initialization().expect("loop owns checked initialization"),&arguments,start_integer,current_integer);
        LoopConstruction{schedule,parent,initial,start,end,carries:carries.to_vec()}
    }

    fn finish_loop(&mut self,ticket:LoopConstruction) {
        let yielded=Bound::Tuple(ticket.carries.iter().map(|carry|self.bound(carry.yielded)).collect());
        let yielded_type=SemanticType::Tuple(ticket.carries.iter().map(|carry|self.function.value(carry.yielded).ty.clone()).collect());
        let yielded=self.realize_region_product(yielded,&yielded_type);
        let backedge=self.region_operand_product(&yielded);
        let iteration=self.values.contents.clone();
        let symbol=ticket.schedule.binding().symbol;
        let destinations=self.builder.finish_value_repeat(ticket.schedule,backedge);
        self.values=ticket.parent;
        let before=self.values.contents.clone();
        let start=self.builder.arena().int_from_nat(ticket.start);
        let end=self.builder.arena().int_from_nat(ticket.end);
        self.values.contents.completed_loop(&mut InitializationContext::new(self.builder.arena()),&before,&iteration,symbol,start,end);
        let Bound::Tuple(results)=self.region_destination_product(&destinations,&yielded,&iteration,Some((&ticket.initial,&before))) else {unreachable!()};
        for (carry,result) in ticket.carries.iter().zip(results) {
            self.values.bind(self.builder.bindings_mut(),carry.result,result);
        }
    }

    fn begin_parallel_segment(
        &mut self,
        start_value: SemanticValueId,
        end_value: SemanticValueId,
        capture_values: &[SemanticValueId],
        body: RegionId,
        carries: &[seismic_lang::entry::Carry],
        _written_places: &[SemanticValueId],
    ) -> construction::SegmentConstruction {
        let start_bound = self.bound(start_value);
        let end_bound = self.bound(end_value);
        let start = index_expr(self.builder.arena(), &start_bound);
        let end = index_expr(self.builder.arena(), &end_bound);
        let max = self.builder.arena().nat_max(end, start);
        let extent = self.builder.arena().nat_sub(max, start);
        let mut helpers = BTreeMap::new();
        let mut intrinsics = Vec::new();
        let mut checks = Vec::new();
        self.collect_segment_contract(
            self.function,
            body,
            &mut helpers,
            &mut intrinsics,
            &mut checks,
        );

        let target = self.builder.portable_target_ref();
        let registry = self.builder.portable_registry_ref();
        let mut required_mode = None;
        let mut required_workgroup = None;
        for intrinsic in &intrinsics {
            let signature = registry::intrinsic_signature(*intrinsic);
            let implementation = registry
                .intrinsic(signature.id)
                .expect("checked intrinsic is absent from the compiler registry");
            let requirements = (implementation.launch_requirements)(
                target,
                self.builder.arena(),
                signature,
                extent,
            );
            merge_exact(
                &mut required_mode,
                requirements.required_mode,
                "launch mode",
            );
            merge_exact(
                &mut required_workgroup,
                requirements.required_workgroup,
                "workgroup geometry",
            );
        }
        let one = self.builder.arena().nat(1);
        let zero = self.builder.arena().nat(0);
        let workgroup = required_workgroup.unwrap_or_else(|| {
            if self.mode == SemanticMode::Portable {
                [one, one, one]
            } else {
                let domain = FiniteDomain::new(vec![1, 32, 64, 128, 256])
                    .expect("canonical workgroup domain is non-empty");
                let decision = self.builder.decision("semantic workgroup size", domain);
                let selected = self.builder.arena().decision_value(decision);
                let selected = self.builder.arena().nat_from_int(selected);
                [selected, one, one]
            }
        });
        let participants = self.builder.arena().nat_product(&workgroup);
        let groups = self.builder.arena().nat_ceil_div(extent, participants);
        let empty = self
            .builder
            .arena()
            .nat_cmp(seismic_lang::expr::CmpOp::Eq, extent, zero);
        let domain = SegmentLaunchDomain {
            mode: required_mode.unwrap_or(LaunchParticipation::Independent),
            grid: [groups, one, one],
            workgroup,
            empty,
            parallel_extent: extent,
        };
        for intrinsic in &intrinsics {
            let participation = registry::intrinsic_signature(*intrinsic)
                .effects
                .participation;
            let cohort = match participation {
                registry::IntrinsicParticipation::Independent => None,
                registry::IntrinsicParticipation::FullSubgroup => None,
                registry::IntrinsicParticipation::FullWorkgroup => Some(participants),
            };
            if let Some(cohort) = cohort {
                let remainder = self.builder.arena().nat_rem(extent, cohort);
                let exact =
                    self.builder
                        .arena()
                        .nat_cmp(seismic_lang::expr::CmpOp::Eq, remainder, zero);
                self.builder.constrain(exact);
            }
        }

        let parameters = self.function.region(body).parameters();
        let RegionKind::LoopBody { binder_value, .. } = self.function.region(body).kind() else {
            panic!("parallel loop body has wrong region kind")
        };
        let capture_parameters = &parameters[1..];
        assert_eq!(capture_parameters.len(), capture_values.len());
        let written_parameters = Self::region_written_parameters(self.function, body);
        let mut outer_captures = Vec::with_capacity(capture_values.len());
        for (value, parameter) in capture_values.iter().zip(capture_parameters) {
            let bound = self.bound(*value);
            outer_captures
                .push(self.segment_capture_plan(&bound, written_parameters.contains(parameter)));
        }
        let function = self.function;
        let program = self.builder.portable_program_ref();
        let compiler_owned_base = self.mode != SemanticMode::AuthoredBackend;
        let check_statuses = self.builder.portable_source_statuses(checks.len());
        let mut kernel = self.builder.portable_kernel();
        let mut values = BTreeMap::new();
        for (parameter, capture) in capture_parameters.iter().zip(&outer_captures) {
            values.insert(*parameter, segment_capture(&mut kernel, capture));
        }
        let start_arg = kernel.nat_arg(start);
        let (global, logical_base) =
            logical_global_id(&mut kernel, compiler_owned_base, domain.parallel_extent);
        let binder = kernel.binary(BinaryOp::Add, start_arg, global);
        values.insert(*binder_value, SegmentBound::Scalar(binder));
        let check_tensors = segment_check_statuses(&mut kernel, &checks, &check_statuses);
        let cohort_scope = cohort::Scope::region(function, body, &helpers);
        let cohort = cohort_scope.map(|_| cohort::Cohort::new(&mut kernel, domain));
        let active = if let Some(scope) = cohort_scope {
            cohort.as_ref().unwrap().membership(&mut kernel, scope, domain.parallel_extent, logical_base)
        } else {
            let end_arg = kernel.nat_arg(end);
            kernel.cmp(CmpOp::Lt, binder, end_arg)
        };
        let branch = kernel.begin_branch(active);
        let alive = kernel.constant(ConstantValue::Bool(true), ValueType::Bool);
        let mut segment = SegmentLowerer {
            function, program, kernel: &mut kernel, values, helpers: &helpers, checks: &check_tensors,
            target, registry, domain, alive, cohort: cohort.as_ref(), successful: uniformity::Values::new(), lexical: registry::IntrinsicUniformity::Workgroup,
        };
        segment.initialize_successful_values(body);
        let environment = segment.into_environment();
        assert!(carries.is_empty(), "checked parallel loops cannot carry reassigned state");
        construction::SegmentConstruction {
            capacity_pending: None,            cursor: kernel.suspend(), environment, branch,
            frames: vec![source_control::SegmentFrame::region(body)],
            helpers: helpers.into_iter().map(|(family, function)| (family, function.id())).collect(),
            checks: check_tensors, cohort, domain, logical_base,
            completion: construction::SegmentCompletion { checks, check_statuses, start, end, body, captures: capture_values.to_vec() },
        }
    }

    fn finish_parallel_segment(&mut self, kernel: seismic_ir::kernel::KernelId, domain: SegmentLaunchDomain, logical_base: Option<seismic_ir::kernel::dynamic::LogicalIndexBinding>, completion: construction::SegmentCompletion) {
        let construction::SegmentCompletion { checks, check_statuses, start, end, body, captures } = completion;
        self.launch_semantic(kernel, domain, logical_base);
        source_scalar::finish(self.builder, self.function.name(), checks, check_statuses);
        let mut loop_values = vec![Bound::Scalar(ScalarBinding::Index(start))];
        loop_values.extend(captures.iter().map(|value| self.bound(*value)));
        let arguments = self.initialization_arguments(&loop_values);
        let start_integer = self.builder.arena().int_from_nat(start);
        let end_integer = self.builder.arena().int_from_nat(end);
        let mut context = initialization_context(self.builder.arena(), &self.values.selections, &self.values.binders);
        self.values.contents.loop_exit(&mut context, self.function.region(body).loop_initialization().expect("loop owns checked initialization"), &arguments, start_integer, end_integer);
    }

    /// Finds loop-body parameters whose storage is written by the body's
    /// authoritative leaf events. Control nodes intentionally do not copy
    /// child events, so write admission must recurse through child regions
    /// instead of consulting the loop wrapper.
    fn region_written_parameters(
        function: &SemanticFunction,
        region: RegionId,
    ) -> BTreeSet<SemanticValueId> {
        let mut written = BTreeSet::new();
        for (_, node) in function.nodes(region) {
            for event in node.events() {
                if matches!(
                    event.kind(),
                    seismic_lang::entry::SemanticEventKind::Write(_)
                        | seismic_lang::entry::SemanticEventKind::AtomicRmw { .. }
                ) {
                    if let Some(parameter) = event.place().and_then(|place| {
                        Self::region_storage_parameter(
                            function,
                            region,
                            place,
                            &mut BTreeSet::new(),
                        )
                    }) {
                        written.insert(parameter);
                    }
                }
            }
            let children: Vec<(RegionId, &[SemanticValueId])> = match node.view() {
                SemanticNodeView::If {
                    captures,
                    then,
                    otherwise,
                    ..
                } => vec![(then, captures), (otherwise, captures)],
                SemanticNodeView::Loop { captures, body, .. } => vec![(body, captures)],
                _ => Vec::new(),
            };
            for (child, captures) in children {
                let child_parameters = function.region(child).parameters();
                let capture_parameters = match function.region(child).kind() {
                    RegionKind::LoopBody { .. } => &child_parameters[1..],
                    RegionKind::Then | RegionKind::Else => child_parameters,
                    RegionKind::Root => panic!("control child region cannot be a root"),
                };
                for child_parameter in Self::region_written_parameters(function, child) {
                    let Some(ordinal) = capture_parameters
                        .iter()
                        .position(|parameter| *parameter == child_parameter)
                    else {
                        continue;
                    };
                    if let Some(parameter) = Self::region_storage_parameter(
                        function,
                        region,
                        captures[ordinal],
                        &mut BTreeSet::new(),
                    ) {
                        written.insert(parameter);
                    }
                }
            }
        }
        written
    }

    fn region_storage_parameter(
        function: &SemanticFunction,
        region: RegionId,
        value: SemanticValueId,
        seen: &mut BTreeSet<SemanticValueId>,
    ) -> Option<SemanticValueId> {
        if !seen.insert(value) {
            return None;
        }
        if function.region(region).parameters().contains(&value) {
            return Some(value);
        }
        let ValueOrigin::Node(node) = function.value(value).origin else {
            return None;
        };
        if node.region() != region {
            return None;
        }
        let base = match function.node(node).view() {
            SemanticNodeView::View { base, .. } => Some(base),
            SemanticNodeView::ElementWrite { place, .. }
            | SemanticNodeView::Atomic { place, .. } => Some(place),
            SemanticNodeView::Store { destination, .. } => Some(destination),
            _ => None,
        }?;
        Self::region_storage_parameter(function, region, base, seen)
    }

    fn collect_segment_contract(
        &self,
        function: &'f SemanticFunction,
        region: RegionId,
        helpers: &mut BTreeMap<FamilyId, &'f SemanticFunction>,
        intrinsics: &mut Vec<IntrinsicId>,
        checks: &mut SourceChecks,
    ) {
        for (id, node) in function.nodes(region) {
            match node.view() {
                SemanticNodeView::Intrinsic { intrinsic, .. } => {
                    assert!(matches!(
                        registry::intrinsic_signature(intrinsic).execution,
                        registry::IntrinsicExecution::WithinEnclosingParallel
                    ));
                    intrinsics.push(intrinsic);
                }
                SemanticNodeView::Call { family, .. } => {
                    let helper = self.builder.segment_callee(family, B::NAME);
                    if helpers.insert(family, helper).is_none() {
                        self.collect_segment_contract(
                            helper,
                            helper.root(),
                            helpers,
                            intrinsics,
                            checks,
                        );
                    }
                }
                SemanticNodeView::Primitive { primitive, inputs, .. }
                | SemanticNodeView::Elementwise { primitive, inputs, .. } => {
                    for check in source_scalar::checks(function,id,primitive,inputs) {
                        if !checks.iter().any(|(site,_)|*site==check.0) { checks.push(check); }
                    }
                }
                SemanticNodeView::Check { reason, .. } => {
                    let site = SourceFailure::at(function, id, SourceFailureCause::Check(reason.clone()));
                    if !checks.iter().any(|(existing, _)| *existing == site) {
                        checks.push((site, node.span()));
                    }
                }
                SemanticNodeView::If {
                    then, otherwise, ..
                } => {
                    self.collect_segment_contract(function, then, helpers, intrinsics, checks);
                    self.collect_segment_contract(function, otherwise, helpers, intrinsics, checks);
                }
                SemanticNodeView::Loop { body, .. } => {
                    self.collect_segment_contract(function, body, helpers, intrinsics, checks);
                }
                _ => {}
            }
        }
    }

    fn segment_capture_plan(&mut self, bound: &Bound, writable: bool) -> SegmentCapture {
        match bound {
            Bound::Tensor(TensorRealization::Stored(value)) => SegmentCapture::Tensor {
                contract: value.clone(),
                writable,
            },
            Bound::Tensor(TensorRealization::Computed(_)) => unreachable!("segment captures are materialized"),
            Bound::Scalar(_) => SegmentCapture::Scalar(self.prepared(bound)),
            Bound::Range { start, end } => SegmentCapture::Range {
                start: Box::new(self.segment_capture_plan(start, false)),
                end: Box::new(self.segment_capture_plan(end, false)),
            },
            Bound::Tuple(items) => SegmentCapture::Tuple(
                items
                    .iter()
                    .map(|item| self.segment_capture_plan(item, writable))
                    .collect(),
            ),
            Bound::Unit => SegmentCapture::Unit,
        }
    }

    fn bind_region_parameters(&mut self, region: RegionId, captures: &[Bound]) {
        let region = self.function.region(region);
        let parameters = match region.kind() {
            RegionKind::LoopBody { binder_value, .. } => {
                let (binder, captures) = region
                    .parameters()
                    .split_first()
                    .expect("loop region is missing its binder parameter");
                assert_eq!(
                    binder, binder_value,
                    "loop binder is not the first region parameter"
                );
                captures
            }
            RegionKind::Root | RegionKind::Then | RegionKind::Else => region.parameters(),
        };
        assert_eq!(
            parameters.len(),
            captures.len(),
            "region capture arity differs from semantic inputs"
        );
        for (parameter, capture) in parameters.iter().zip(captures) {
            self.values
                .bind(self.builder.bindings_mut(), *parameter, capture.clone());
        }
    }
    fn publication_selection(&mut self) -> Option<BindingSelector> {
        if !self.builder.portable_is_root() { return None; }
        // Scalar publication slots are stable across selected arms. Tensor
        // publication consumes each arm's actual completed descriptor instead
        // of allocating a nominal destination before the arm executes.
        for result in self.function.results() {
            if !matches!(self.function.value(*result).ty, SemanticType::Tensor(_)) {
                let _ = self.output_target(*result);
            }
        }
        self.function.results().iter().find_map(|value| {
            self.builder
                .bindings()
                .unresolved(self.values.handle(*value), &self.values.selections)
        })
    }

    fn publish_results(&mut self) {
        if !self.builder.portable_is_root() {
            self.builder.bindings_mut().result_contents = self.values.contents.clone();
            self.builder.bindings_mut().results = self
                .function
                .results()
                .iter()
                .map(|result| self.values.handle(*result))
                .collect();
            return;
        }
        assert!(self.publication_selection().is_none(), "publication requires its selected construction arm");
        for result in self.function.results() {
            self.materialize_tensor(*result);
            let source = self.bound(*result);
            let target = if matches!(source, Bound::Tensor(_)) {
                let ty = self.function.value(*result).ty.clone();
                let realized = self.realize_region_product(source, &ty);
                let view = *realized.tensor().direct_backing()
                    .expect("publication realization must have an affine descriptor");
                self.builder.portable_publish_tensor(*result, view);
                realized
            } else {
                let target = self.output_target(*result);
                self.assign(&source, &target);
                target
            };
            self.values.rebind(self.builder.bindings_mut(), *result, target);
        }
        self.builder.bindings_mut().result_contents = self.values.contents.clone();
        self.builder.bindings_mut().results = self
            .function
            .results()
            .iter()
            .map(|result| self.values.handle(*result))
            .collect();
    }
}

struct SegmentLowerer<'s, 'k, 'f, 'r, B: seismic_target::TargetFamily> {
    function: &'f SemanticFunction,
    program: &'f seismic_lang::entry::SemanticProgram,
    kernel: &'s mut PortableBuilder<'k, B>,
    values: BTreeMap<SemanticValueId, SegmentBound>,
    helpers: &'r BTreeMap<FamilyId, &'f SemanticFunction>,
    checks: &'r SourceStatuses,
    target: &'f seismic_target::DeviceDescription<B>,
    registry: &'f crate::target::CompilerRegistry<B>,
    domain: SegmentLaunchDomain,
    alive: PortableValue,
    cohort: Option<&'r cohort::Cohort>,
    successful: uniformity::Values,
    lexical: registry::IntrinsicUniformity,
}

impl<'s, 'k, 'f, 'r, B: seismic_target::TargetFamily> SegmentLowerer<'s, 'k, 'f, 'r, B> {
    fn bound(&self, value: SemanticValueId) -> SegmentBound {
        self.values
            .get(&value)
            .cloned()
            .unwrap_or_else(|| panic!("checked segment value is not dominated"))
    }

    fn scalar(&self, value: SemanticValueId) -> PortableValue {
        match self.bound(value) {
            SegmentBound::Scalar(value) => value,
            _ => panic!("checked segment scalar has non-scalar binding"),
        }
    }

    fn tensor(&self, value: SemanticValueId) -> SegmentTensor {
        match self.bound(value) {
            SegmentBound::Tensor(value) => value,
            _ => panic!("checked segment tensor has non-tensor binding"),
        }
    }

    fn physical_tensor(&mut self, value: SemanticValueId) -> PortableTensor {
        fn place<B: seismic_target::TargetFamily>(
            kernel: &mut PortableBuilder<'_, B>,
            tensor: SegmentTensor,
        ) -> PortableTensor {
            match tensor.value {
                TensorDefinitionValue::Physical(value) => value.tensor,
                TensorDefinitionValue::View { base, transform } => {
                    let base_rank = base.axes.len();
                    let base = place(kernel, (*base).clone());
                    match transform {
                        SegmentViewTransform::Transpose(permutation) => {
                            kernel.tensor_transpose(base, permutation)
                        }
                        SegmentViewTransform::Reshape => kernel.tensor_reshape(base, tensor.axes),
                        SegmentViewTransform::Slice(axes) => {
                            let mut output_axes = tensor.axes.iter();
                            let mut selections = axes
                                .into_iter()
                                .map(|axis| match axis {
                                    SegmentSliceAxis::Point(value) => {
                                        PortableSliceAxis::Point(value)
                                    }
                                    SegmentSliceAxis::Full => {
                                        output_axes.next().expect("full slice retains an axis");
                                        PortableSliceAxis::Full
                                    }
                                    SegmentSliceAxis::Range { start } => {
                                        let extent = *output_axes
                                            .next()
                                            .expect("range slice retains an axis");
                                        let end = kernel.binary(BinaryOp::Add, start, extent);
                                        PortableSliceAxis::Range { start, end }
                                    }
                                })
                                .collect::<Vec<_>>();
                            assert!(
                                selections.len() <= base_rank,
                                "slice exceeds its source rank"
                            );
                            selections.resize(base_rank, PortableSliceAxis::Full);
                            kernel.tensor_slice(base, selections)
                        }
                    }
                }
                _ => panic!("checked writable/capability tensor is not physical storage"),
            }
        }
        let tensor = self.tensor(value);
        place(self.kernel, tensor)
    }

    fn lower_node(&mut self, id: NodeId, result: Option<producer::TensorResult<PortableValue>>) {
        let node = self.function.node(id);
        match node.view() {
            SemanticNodeView::Primitive {
                primitive,
                inputs,
                output,
            } => self.lower_primitive(id, primitive, inputs, output),
            SemanticNodeView::Intrinsic {
                intrinsic,
                inputs,
                output,
            } => self.lower_intrinsic(intrinsic, inputs, output),
            SemanticNodeView::Elementwise {
                primitive,
                inputs,
                output,
            } => self.lower_elementwise(id, primitive, inputs, output, result.unwrap()),
            SemanticNodeView::Reduce {
                op,
                axis,
                input,
                output,
                ..
            } => self.lower_reduce(op, axis, input, output, result.unwrap()),
            SemanticNodeView::View {
                base,
                transform,
                extents,
                output,
            } => {
                let mut tensor = self.tensor(base);
                tensor = match transform {
                    ViewTransform::Identity => tensor,
                    ViewTransform::Plane { plane } => {
                        let TensorDefinitionValue::Physical(physical) = tensor.value else {
                            panic!("checked plane view requires packed physical storage")
                        };
                        let physical = self.kernel.tensor_plane(physical.tensor, *plane);
                        SegmentTensor::physical(self.kernel, SegmentStorage::source(result.unwrap(), physical))
                    }
                    ViewTransform::Transpose { permutation } => SegmentTensor {
                        axes: permutation
                            .iter()
                            .map(|axis| tensor.axes[*axis as usize])
                            .collect(),
                        value: TensorDefinitionValue::View {
                            base: Arc::new(tensor),
                            transform: SegmentViewTransform::Transpose(permutation.clone()),
                        },
                    },
                    ViewTransform::Reshape { .. } => {
                        let axes = extents.iter().map(|value| { let value=self.scalar(*value); portable_index(self.kernel,value) }).collect();
                        SegmentTensor {
                            axes,
                            value: TensorDefinitionValue::View {
                                base: Arc::new(tensor),
                                transform: SegmentViewTransform::Reshape,
                            },
                        }
                    }
                    ViewTransform::Slice { axes } => {
                        let mut mapped = Vec::with_capacity(axes.len());
                        let mut output_axes = Vec::new();
                        for (axis, selection) in axes.iter().enumerate() {
                            mapped.push(match selection {
                                SliceAxis::Full => {
                                    output_axes.push(tensor.axes[axis]);
                                    SegmentSliceAxis::Full
                                }
                                SliceAxis::Point { value, .. } => {
                                    SegmentSliceAxis::Point(self.scalar_ref(value))
                                }
                                SliceAxis::Range { start, end, .. } => {
                                    let start = start
                                        .as_ref()
                                        .map(|value| self.scalar_ref(value))
                                        .unwrap_or_else(|| self.kernel.index_constant(0));
                                    let end = end
                                        .as_ref()
                                        .map(|value| self.scalar_ref(value))
                                        .unwrap_or(tensor.axes[axis]);
                                    output_axes.push(self.kernel.binary(BinaryOp::Sub, end, start));
                                    SegmentSliceAxis::Range { start }
                                }
                            });
                        }
                        output_axes.extend_from_slice(&tensor.axes[axes.len()..]);
                        SegmentTensor {
                            axes: output_axes,
                            value: TensorDefinitionValue::View {
                                base: Arc::new(tensor),
                                transform: SegmentViewTransform::Slice(mapped),
                            },
                        }
                    }
                };
                self.values.insert(output, SegmentBound::Tensor(tensor));
            }
            SemanticNodeView::ElementRead {
                place,
                indices,
                output,
            } => {
                let tensor = self.tensor(place);
                let indices = indices
                    .iter()
                    .map(|value| {
                        let value = self.scalar(*value);
                        portable_index(self.kernel, value)
                    })
                    .collect::<Vec<_>>();
                let value = tensor.read(self.kernel, &indices);
                self.values.insert(output, SegmentBound::Scalar(value));
            }
            SemanticNodeView::ElementWrite {
                place,
                indices,
                value,
                output,
            } => {
                let tensor = self.physical_tensor(place);
                let indices = indices
                    .iter()
                    .map(|value| {
                        let value = self.scalar(*value);
                        portable_index(self.kernel, value)
                    })
                    .collect::<Vec<_>>();
                let mut value = self.scalar(value);
                let SemanticType::Tensor(tensor_type) = &self.function.value(place).ty else {
                    panic!("checked element write place is not a tensor")
                };
                let destination_type = value_type(element_dtype(tensor_type.representation));
                if value.ty() != destination_type {
                    value = self.kernel.cast(value, destination_type);
                }
                self.kernel.tensor_write(&tensor, &indices, value);
                self.values.insert(
                    output,
                    self.bound(place),
                );
            }
            SemanticNodeView::Store {
                destination,
                value,
                output,
            } => self.lower_store(destination, value, output, result.unwrap()),
            SemanticNodeView::Atomic {
                op,
                place,
                arguments,
                output,
            } => {
                let tensor = self.physical_tensor(place);
                let Some((&value, indices)) = arguments.split_last() else {
                    panic!("checked atomic is missing its value")
                };
                let indices = indices
                    .iter()
                    .map(|value| {
                        let value = self.scalar(*value);
                        portable_index(self.kernel, value)
                    })
                    .collect::<Vec<_>>();
                self.kernel
                    .tensor_atomic(op, &tensor, &indices, self.scalar(value));
                self.values.insert(
                    output,
                    self.bound(place),
                );
            }
            SemanticNodeView::Extent {
                tensor,
                axis,
                output,
            } => {
                let tensor = self.tensor(tensor);
                let value = tensor.axes[axis as usize];
                self.values.insert(output, SegmentBound::Scalar(value));
            }
            SemanticNodeView::TuplePack { inputs, output } => {
                let values = inputs.iter().map(|value| self.bound(*value)).collect();
                self.values.insert(output, SegmentBound::Tuple(values));
            }
            SemanticNodeView::TupleGet {
                tuple,
                index,
                output,
            } => {
                let SegmentBound::Tuple(values) = self.bound(tuple) else {
                    panic!("checked tuple projection has non-tuple input")
                };
                self.values.insert(output, values[index as usize].clone());
            }
            SemanticNodeView::Call { .. }
            | SemanticNodeView::If { .. }
            | SemanticNodeView::Check { .. } => {
                unreachable!("segment control nodes are consumed by the continuation walker")
            }
            SemanticNodeView::Loop { .. } => unreachable!("segment loops are consumed by their owned continuation"),
            SemanticNodeView::Alloc { output, .. } => self.lower_local_alloc(output, result.unwrap()),
            SemanticNodeView::Fill { value, output, .. } => self.lower_local_fill(value, output, result.unwrap()),
            SemanticNodeView::Copy { input, output } => self.lower_local_copy(input, output, result.unwrap()),
            SemanticNodeView::RepresentationConvert { .. } => {
                panic!("representation conversion cannot be nested in a participant segment")
            }
        }
    }

    fn scalar_ref(&mut self, value: &ScalarRef) -> PortableValue {
        let value = match value {
            ScalarRef::Static(value) => self.kernel.nat_arg(*value),
            ScalarRef::Value(value) => self.scalar(*value),
        };
        portable_index(self.kernel, value)
    }

    fn lower_primitive(
        &mut self,
        node: NodeId,
        primitive: &PrimitiveId,
        inputs: &[SemanticValueId],
        output: SemanticValueId,
    ) {
        match primitive {
            PrimitiveId::RangeMake => {
                self.values.insert(
                    output,
                    SegmentBound::Range {
                        start: Box::new(self.bound(inputs[0])),
                        end: Box::new(self.bound(inputs[1])),
                    },
                );
            }
            PrimitiveId::RangeStart | PrimitiveId::RangeEnd => {
                let SegmentBound::Range { start, end } = self.bound(inputs[0]) else {
                    panic!("checked range endpoint has non-range input")
                };
                self.values.insert(
                    output,
                    if matches!(primitive, PrimitiveId::RangeStart) {
                        *start
                    } else {
                        *end
                    },
                );
            }
            PrimitiveId::Symbolic(expression) => {
                let values = self.values.clone();
                let captured = capture_expr_with(
                    self.kernel.expression_arena(),
                    AnyExpr::Int(*expression),
                    &mut |_, value| match values.get(&value) {
                        Some(SegmentBound::Scalar(value)) => CapturedExpr::Value(*value),
                        _ => panic!("symbolic segment value has no scalar binding"),
                    },
                );
                let mut value = lower_captured_expr(self.kernel, &captured);
                if matches!(self.function.value(output).ty, SemanticType::Index { .. })
                    && value.ty() != ValueType::Index
                {
                    value = self.kernel.cast(value, ValueType::Index);
                }
                self.values.insert(output, SegmentBound::Scalar(value));
            }
            _ => {
                let arguments = inputs
                    .iter()
                    .map(|value| self.scalar(*value))
                    .collect::<Vec<_>>();
                let (value, successful) = source_scalar::lower(
                    self.kernel, self.function, node, self.checks, self.alive,
                    primitive, &arguments, &self.function.value(output).ty,
                );
                self.alive = successful;
                self.values.insert(output, SegmentBound::Scalar(value));
            }
        }
    }

    fn lower_elementwise(
        &mut self,
        node: NodeId,
        primitive: &PrimitiveId,
        inputs: &[SemanticValueId],
        output: SemanticValueId,
        result: producer::TensorResult<PortableValue>,
    ) {
        let SemanticType::Tensor(output_type) = &self.function.value(output).ty else {
            panic!("checked elementwise output is not a tensor")
        };
        if !source_scalar::checks(self.function,node,primitive,inputs).is_empty() {
            let storage=self.local_tensor(result);
            let tensor=&storage.tensor;
            let axes=self.kernel.tensor_extents(&tensor).to_vec();
            let arguments=inputs.iter().map(|value|self.bound(*value)).collect::<Vec<_>>();
            let output_type=SemanticType::Scalar(element_dtype(output_type.representation));
            let statuses=self.checks;
            let mut emit=|kernel:&mut PortableBuilder<'_,B>,index:&[PortableValue],alive:PortableValue| {
                kernel.branch(alive,|kernel| {
                    let args=segment_elementwise_arguments(kernel,&arguments,index);
                    let (value,successful)=source_scalar::lower(kernel,self.function,node,statuses,alive,primitive,&args,&output_type);
                    kernel.branch(successful,|kernel| {kernel.tensor_write(&tensor,index,value);Vec::new()},|_|Vec::new());
                    vec![successful]
                },|_|vec![alive])[0]
            };
            self.alive=source_scalar::nested(self.kernel,&axes,0,&mut Vec::new(),self.alive,&mut emit);
            self.values.insert(output,SegmentBound::Tensor(SegmentTensor::physical(self.kernel,storage)));
            return;
        }
        let axes = result.axes(self.kernel);
        let inputs = inputs
            .iter()
            .map(|value| self.bound(*value))
            .collect::<Vec<_>>();
        self.values.insert(
            output,
            SegmentBound::Tensor(SegmentTensor {
                axes,
                value: TensorDefinitionValue::Elementwise {
                    primitive: primitive.clone(),
                    inputs,
                    result,
                },
            }),
        );
    }

    fn lower_reduce(
        &mut self,
        op: ReduceOp,
        axis: u32,
        input: SemanticValueId,
        output: SemanticValueId,
        result: producer::TensorResult<PortableValue>,
    ) {
        let input_id = input;
        let input = self.tensor(input_id);
        let SemanticType::Tensor(input_type) = &self.function.value(input_id).ty else {
            unreachable!()
        };
        let axes = result.axes(self.kernel);
        let reduced = SegmentTensor {
            axes,
            value: TensorDefinitionValue::Reduce {
                op,
                axis: axis as usize,
                input_dtype: registry::representation_info(input_type.representation).decoded,
                input: Arc::new(input),
                result,
            },
        };
        let value = match &self.function.value(output).ty {
            SemanticType::Scalar(_) => {
                assert!(
                    reduced.axes.is_empty(),
                    "scalar reduction retains tensor axes"
                );
                SegmentBound::Scalar(reduced.read(self.kernel, &[]))
            }
            SemanticType::Tensor(_) => SegmentBound::Tensor(reduced),
            _ => unreachable!("checked reduction result is neither scalar nor tensor"),
        };
        self.values.insert(output, value);
    }

    fn lower_store(
        &mut self,
        destination: SemanticValueId,
        value: SemanticValueId,
        output: SemanticValueId,
        result: producer::TensorResult<PortableValue>,
    ) {
        let destination_tensor = self.physical_tensor(destination);
        let mut source = self.tensor(value);
        if source.reads_overlap(self.kernel,std::slice::from_ref(&destination_tensor)) {
            let SemanticType::Tensor(tensor)=&self.function.value(value).ty else {unreachable!("store RHS is a tensor")};
            let snapshot=result.snapshot_storage(self.kernel,&source.axes);
            let dtype=registry::representation_info(tensor.representation).decoded;
            segment_copy_tensor(self.kernel,&source,&snapshot,dtype,&source.axes,0,&mut Vec::new());
            source=SegmentTensor::physical(self.kernel,SegmentStorage::source(result,snapshot));
        }
        let axes = self.kernel.tensor_extents(&destination_tensor).to_vec();
        let SemanticType::Tensor(destination_type) = &self.function.value(destination).ty else {
            unreachable!()
        };
        let dtype = registry::representation_info(destination_type.representation).decoded;
        segment_copy_tensor(
            self.kernel,
            &source,
            &destination_tensor,
            dtype,
            &axes,
            0,
            &mut Vec::new(),
        );
        let SemanticType::Tensor(TensorSemantics {
            storage: TensorStorage::View { base, .. },
            ..
        }) = &self.function.value(destination).ty
        else {
            panic!("checked store destination is not a writable view")
        };
        self.values.insert(output, self.bound(*base));
    }

    fn local_tensor(&mut self, result: producer::TensorResult<PortableValue>) -> SegmentStorage {
        let axes=result.axes(self.kernel);
        let tensor=result.snapshot_storage(self.kernel,&axes);
        SegmentStorage::source(result,tensor)
    }

    fn lower_local_alloc(&mut self, output: SemanticValueId, result: producer::TensorResult<PortableValue>) {
        let storage = self.local_tensor(result);
        self.values.insert(
            output,
            SegmentBound::Tensor(SegmentTensor::physical(self.kernel, storage)),
        );
    }

    fn lower_local_fill(
        &mut self,
        fill: seismic_lang::intrinsics::FillConstant,
        output: SemanticValueId,
        result: producer::TensorResult<PortableValue>,
    ) {
        let storage = self.local_tensor(result);
        let tensor = &storage.tensor;
        let axes = self.kernel.tensor_extents(&tensor).to_vec();
        let dtype = match &self.function.value(output).ty {
            SemanticType::Tensor(tensor) => {
                registry::representation_info(tensor.representation).decoded
            }
            _ => unreachable!(),
        };
        let value = match fill {
            seismic_lang::intrinsics::FillConstant::Zero => zero_of(self.kernel, value_type(dtype)),
            seismic_lang::intrinsics::FillConstant::One => one_of(self.kernel, value_type(dtype)),
        };
        segment_fill_tensor(self.kernel, &tensor, &axes, 0, &mut Vec::new(), value);
        self.values.insert(
            output,
            SegmentBound::Tensor(SegmentTensor::physical(self.kernel, storage)),
        );
    }

    fn lower_local_copy(&mut self, input: SemanticValueId, output: SemanticValueId, result: producer::TensorResult<PortableValue>) {
        let source = self.tensor(input);
        let storage = self.local_tensor(result);
        let tensor = &storage.tensor;
        let axes = self.kernel.tensor_extents(&tensor).to_vec();
        let SemanticType::Tensor(output_type) = &self.function.value(output).ty else {
            unreachable!()
        };
        let dtype = registry::representation_info(output_type.representation).decoded;
        segment_copy_tensor(
            self.kernel,
            &source,
            &tensor,
            dtype,
            &axes,
            0,
            &mut Vec::new(),
        );
        self.values.insert(
            output,
            SegmentBound::Tensor(SegmentTensor::physical(self.kernel, storage)),
        );
    }

    fn lower_intrinsic(
        &mut self,
        intrinsic: IntrinsicId,
        inputs: &[SemanticValueId],
        output: SemanticValueId,
    ) {
        let signature = registry::intrinsic_signature(intrinsic);
        let mut operands = Vec::with_capacity(inputs.len());
        for (input, argument) in inputs.iter().zip(&signature.arguments) {
            operands.push(match argument.category {
                registry::OperandCategory::Scalar(dtype) => {
                    let value = self.scalar(*input);
                    SemanticIntrinsicOperand::Scalar(self.kernel.semantic_scalar(value, dtype))
                }
                registry::OperandCategory::Constant(dtype) => {
                    let value = self.scalar(*input);
                    SemanticIntrinsicOperand::Constant(self.kernel.semantic_scalar(value, dtype))
                }
                registry::OperandCategory::Readable {
                    representation,
                    rank,
                } => {
                    let tensor = self.physical_tensor(*input);
                    SemanticIntrinsicOperand::Readable(self.kernel.semantic_place(
                        tensor,
                        representation,
                        rank,
                        false,
                    ))
                }
                registry::OperandCategory::Writable {
                    representation,
                    rank,
                } => {
                    let tensor = self.physical_tensor(*input);
                    SemanticIntrinsicOperand::Writable(self.kernel.semantic_place(
                        tensor,
                        representation,
                        rank,
                        true,
                    ))
                }
                registry::OperandCategory::Opaque { .. } => match self.bound(*input) {
                    SegmentBound::Opaque(value) => SemanticIntrinsicOperand::Opaque(value),
                    _ => panic!("checked opaque intrinsic operand has non-opaque binding"),
                },
            });
        }
        let call = SemanticIntrinsicCall {
            signature,
            operands: &operands,
            destination: None,
        };
        let mut sink = SemanticIntrinsicSink::open(self.kernel, &call);
        let implementation = self
            .registry
            .intrinsic(call.signature.id)
            .expect("checked intrinsic is absent from the compiler registry");
        (implementation.lower)(self.target, &self.domain, call, &mut sink);
        let result = sink.finish();
        let value = match result {
            SemanticIntrinsicResult::Scalar(value) => SegmentBound::Scalar(value.value()),
            SemanticIntrinsicResult::Opaque(value) => SegmentBound::Opaque(value),
            SemanticIntrinsicResult::Void => SegmentBound::Unit,
            SemanticIntrinsicResult::Owned(_) => {
                panic!("enclosing-parallel intrinsic cannot return an owned tensor")
            }
        };
        self.values.insert(output, value);
    }


}

#[derive(Clone, Copy, Debug)]
pub(crate) enum PreparedArg {
    Index(NatExpr),
    Integer(seismic_lang::expr::IntExpr),
    Scalar(seismic_lang::expr::SymbolId, DType),
}

#[derive(Clone)]
enum IntrinsicPrepared {
    Scalar(PreparedArg, DType, bool),
    Place(
        StoredView,
        seismic_lang::ids::RepresentationId,
        u32,
        bool,
    ),
}


fn merge_exact<T: Copy + PartialEq>(slot: &mut Option<T>, value: Option<T>, what: &str) {
    let Some(value) = value else { return };
    match slot {
        Some(existing) => assert!(
            *existing == value,
            "incompatible {what} requirements in one semantic segment"
        ),
        None => *slot = Some(value),
    }
}

fn segment_capture<B: seismic_target::TargetFamily>(
    kernel: &mut PortableBuilder<'_, B>,
    bound: &SegmentCapture,
) -> SegmentBound {
    match bound {
        SegmentCapture::Tensor { contract, writable } => {
            let tensor = stored_view_in_kernel(kernel, &contract.view, *writable);
            SegmentBound::Tensor(SegmentTensor::physical(kernel, SegmentStorage { tensor, origin: StorageOrigin::Parameter(contract.clone()) }))
        }
        SegmentCapture::Scalar(argument) => SegmentBound::Scalar(match argument {
            PreparedArg::Index(expression) => kernel.nat_arg(*expression),
            PreparedArg::Integer(_) => panic!("unbounded Integer cannot enter a native segment"),
            PreparedArg::Scalar(symbol, dtype) => kernel.scalar_arg(*symbol, *dtype),
        }),
        SegmentCapture::Range { start, end } => SegmentBound::Range {
            start: Box::new(segment_capture(kernel, start)),
            end: Box::new(segment_capture(kernel, end)),
        },
        SegmentCapture::Tuple(items) => SegmentBound::Tuple(
            items
                .iter()
                .map(|item| segment_capture(kernel, item))
                .collect(),
        ),
        SegmentCapture::Unit => SegmentBound::Unit,
    }
}

fn bind_segment_region_parameters(
    values: &mut BTreeMap<SemanticValueId, SegmentBound>,
    function: &SemanticFunction,
    region: RegionId,
    captures: &[SegmentBound],
) {
    let parameters = function.region(region).parameters();
    assert_eq!(parameters.len(), captures.len());
    for (parameter, capture) in parameters.iter().zip(captures) {
        values.insert(*parameter, capture.clone());
    }
}

fn segment_check_statuses<B: seismic_target::TargetFamily>(
    kernel: &mut PortableBuilder<'_, B>,
    checks: &[(SourceFailure, seismic_lang::span::Span)],
    statuses: &[crate::implementation::PortableSourceStatus],
) -> SourceStatuses {
    assert_eq!(checks.len(), statuses.len());
    let Some(first) = statuses.first() else {
        return BTreeMap::new();
    };
    let place = kernel.arg_view(first.view, true);
    let tensor = kernel.tensor(place);
    checks
        .iter()
        .zip(statuses)
        .map(|((id, _), status)| {
            assert_eq!(
                status.view, first.view,
                "one segment owns one status allocation"
            );
            (id.clone(), (tensor.clone(), kernel.index_constant(status.index)))
        })
        .collect()
}

fn segment_elementwise_arguments<B:seismic_target::TargetFamily>(
    kernel:&mut PortableBuilder<'_,B>,inputs:&[SegmentBound],index:&[PortableValue]) -> Vec<PortableValue> {
    inputs.iter().map(|input|match input {
        SegmentBound::Scalar(value)=>*value,
        SegmentBound::Tensor(tensor)=> {
            let skip=index.len()-tensor.axes.len();
            let one=kernel.index_constant(1);
            let zero=kernel.index_constant(0);
            let coordinates=index[skip..].iter().zip(&tensor.axes).map(|(value,extent)| {
                let broadcast=kernel.cmp(CmpOp::Eq,*extent,one);
                kernel.select(broadcast,zero,*value)
            }).collect::<Vec<_>>();
            tensor.read(kernel,&coordinates)
        }
        _=>panic!("elementwise expression input is not scalar or tensor"),
    }).collect()
}

fn segment_copy_tensor<B: seismic_target::TargetFamily>(
    kernel: &mut PortableBuilder<'_, B>,
    source: &SegmentTensor,
    destination: &PortableTensor,
    destination_dtype: DType,
    axes: &[PortableValue],
    axis: usize,
    index: &mut Vec<PortableValue>,
) {
    if axis == axes.len() {
        let value = source.read(kernel, index);
        let ty = value_type(destination_dtype);
        let value = if value.ty() == ty {
            value
        } else {
            kernel.cast(value, ty)
        };
        kernel.tensor_write(destination, index, value);
        return;
    }
    let zero = kernel.index_constant(0);
    let end = axes[axis];
    kernel.repeat(zero, end, Vec::new(), &[], |kernel, binder, _| {
        index.push(binder);
        segment_copy_tensor(
            kernel,
            source,
            destination,
            destination_dtype,
            axes,
            axis + 1,
            index,
        );
        index.pop();
        Vec::new()
    });
}

fn segment_fill_tensor<B: seismic_target::TargetFamily>(
    kernel: &mut PortableBuilder<'_, B>,
    destination: &PortableTensor,
    axes: &[PortableValue],
    axis: usize,
    index: &mut Vec<PortableValue>,
    value: PortableValue,
) {
    if axis == axes.len() {
        kernel.tensor_write(destination, index, value);
        return;
    }
    let zero = kernel.index_constant(0);
    let end = axes[axis];
    kernel.repeat(zero, end, Vec::new(), &[], |kernel, binder, _| {
        index.push(binder);
        segment_fill_tensor(kernel, destination, axes, axis + 1, index, value);
        index.pop();
        Vec::new()
    });
}

fn unravel_index<B: seismic_target::TargetFamily>(
    kernel: &mut PortableBuilder<'_, B>,
    mut linear: PortableValue,
    axes: &[PortableValue],
) -> Vec<PortableValue> {
    let zero = kernel.index_constant(0);
    let mut index = vec![zero; axes.len()];
    for axis in (0..axes.len()).rev() {
        index[axis] = kernel.binary(BinaryOp::Rem, linear, axes[axis]);
        linear = kernel.binary(BinaryOp::Div, linear, axes[axis]);
    }
    index
}

fn value_type(dtype: DType) -> ValueType {
    if dtype == DType::Bool {
        ValueType::Bool
    } else {
        ValueType::Scalar(dtype)
    }
}
fn element_dtype(representation: seismic_lang::ids::RepresentationId) -> DType {
    match registry::representation_info(representation).kind {
        RepresentationKind::Dense(dtype) => dtype,
        RepresentationKind::Packed(_) => DType::F32,
        RepresentationKind::External(_) => {
            panic!("external artifact representation has no element-read dtype")
        }
    }
}
fn representation_unit_bytes(representation: seismic_lang::ids::RepresentationId) -> u64 {
    match registry::representation_info(representation).kind {
        RepresentationKind::Dense(dtype) => u64::from(dtype.bytes()),
        RepresentationKind::Packed(ref packet) => u64::from(packet.packet_size),
        RepresentationKind::External(ref packet) => u64::from(packet.packet_size),
    }
}

fn dense_strides(
    arena: &mut ExprArena,
    representation: seismic_lang::ids::RepresentationId,
    axes: &[NatExpr],
) -> Vec<NatExpr> {
    assert!(
        matches!(
            registry::representation_info(representation).kind,
            RepresentationKind::Dense(_)
        ),
        "reshape of packed representation is forbidden by checking"
    );
    let mut stride = arena.nat(1);
    let mut strides = vec![stride; axes.len()];
    for axis in (0..axes.len()).rev() {
        strides[axis] = stride;
        stride = arena.nat_mul(stride, axes[axis]);
    }
    strides
}

fn prepare_scalar(arena: &mut ExprArena, value: ScalarBinding) -> PreparedArg {
    match value {
        ScalarBinding::Integer(value) => PreparedArg::Integer(value),
        ScalarBinding::Index(value) => PreparedArg::Index(value),
        ScalarBinding::Quantity(slot) => match slot.kind() {
            seismic_ir::schedule::HostQuantityKind::Natural => {
                PreparedArg::Index(arena.nat_symbol(slot.symbol()))
            }
            seismic_ir::schedule::HostQuantityKind::Integer => {
                PreparedArg::Integer(arena.int_symbol(slot.symbol()))
            }
        },
        ScalarBinding::Value { symbol, dtype } => PreparedArg::Scalar(symbol, dtype),
        ScalarBinding::Published(slot) => match slot.kind() {
            seismic_ir::repr::ScalarKind::Nat64 => {
                PreparedArg::Index(arena.nat_symbol(slot.symbol()))
            }
            seismic_ir::repr::ScalarKind::Scalar(dtype) => {
                PreparedArg::Scalar(slot.symbol(), dtype)
            }
        },
    }
}

/// Read the actual bound source value as a mathematical integer. A native word
/// enters with its current typed bits; a quantity keeps its exact host value.
fn host_integer(arena: &mut ExprArena, bound: &Bound) -> IntExpr {
    match prepare_scalar(arena, bound.scalar()) {
        PreparedArg::Integer(value) => value,
        PreparedArg::Index(value) => arena.int_from_nat(value),
        PreparedArg::Scalar(symbol, DType::I32) => {
            let value = arena.scalar_symbol::<seismic_lang::expr::I32>(symbol);
            arena.int_from_scalar(value)
        }
        PreparedArg::Scalar(symbol, DType::U32) => {
            let value = arena.scalar_symbol::<seismic_lang::expr::U32>(symbol);
            arena.int_from_scalar(value)
        }
        _ => panic!("checked integer operation has no integer operand"),
    }
}

fn condition_expr(arena: &mut ExprArena, bound: &Bound) -> seismic_lang::expr::BoolExpr {
    let PreparedArg::Scalar(symbol, DType::Bool) = prepare_scalar(arena, bound.scalar()) else {
        panic!("checked condition is not Boolean")
    };
    let value = arena.scalar_symbol::<seismic_lang::expr::BoolScalar>(symbol);
    let yes = arena.scalar_const::<seismic_lang::expr::BoolScalar>(true);
    arena.scalar_cmp(seismic_lang::expr::CmpOp::Eq, value, yes)
}
fn index_expr(arena: &mut ExprArena, bound: &Bound) -> NatExpr {
    let PreparedArg::Index(value) = prepare_scalar(arena, bound.scalar()) else {
        panic!("checked index has a scalar realization")
    };
    value
}

fn nested<B: seismic_target::TargetFamily>(
    kernel: &mut PortableBuilder<'_, B>,
    axes: &[NatExpr],
    axis: usize,
    index: &mut Vec<PortableValue>,
    body: &mut dyn FnMut(&mut PortableBuilder<'_, B>, &[PortableValue]),
) {
    if axis == axes.len() {
        body(kernel, index);
        return;
    }
    let start = kernel.index_constant(0);
    let end = kernel.nat_arg(axes[axis]);
    kernel.repeat(start, end, Vec::new(), &[], |kernel, binder, _| {
        index.push(binder);
        nested(kernel, axes, axis + 1, index, body);
        index.pop();
        Vec::new()
    });
}
fn reduction_index(
    outer: &[PortableValue],
    axis: usize,
    reduction: PortableValue,
) -> Vec<PortableValue> {
    let mut out = outer.to_vec();
    out.insert(axis, reduction);
    out
}
fn zero_of<B: seismic_target::TargetFamily>(
    kernel: &mut PortableBuilder<'_, B>,
    ty: ValueType,
) -> PortableValue {
    match ty {
        ValueType::Scalar(DType::F32) => kernel.constant(ConstantValue::F32(0.0), ty),
        ValueType::Scalar(DType::I32) => kernel.constant(ConstantValue::I32(0), ty),
        ValueType::Scalar(DType::U32) => kernel.constant(ConstantValue::U32(0), ty),
        ValueType::Scalar(DType::Bool) => kernel.constant(ConstantValue::Bool(false), ty),
        ValueType::Bool => kernel.constant(ConstantValue::Bool(false), ty),
        ValueType::Index => kernel.index_constant(0),
        ValueType::Scalar(DType::F16) => kernel.constant(ConstantValue::F16(0), ty),
        ValueType::Scalar(DType::BF16) => kernel.constant(ConstantValue::BF16(0), ty),
        ValueType::Vector { .. } | ValueType::Opaque { .. } => panic!("non-scalar accumulator"),
    }
}

fn one_of<B: seismic_target::TargetFamily>(
    kernel: &mut PortableBuilder<'_, B>,
    ty: ValueType,
) -> PortableValue {
    match ty {
        ValueType::Scalar(DType::F32) => kernel.constant(ConstantValue::F32(1.0), ty),
        ValueType::Scalar(DType::F16) => {
            kernel.constant(ConstantValue::F16(0x3c00), ValueType::Scalar(DType::F16))
        }
        ValueType::Scalar(DType::BF16) => {
            kernel.constant(ConstantValue::BF16(0x3f80), ValueType::Scalar(DType::BF16))
        }
        ValueType::Scalar(DType::I32) => kernel.constant(ConstantValue::I32(1), ty),
        ValueType::Scalar(DType::U32) => kernel.constant(ConstantValue::U32(1), ty),
        ValueType::Scalar(DType::Bool) => kernel.constant(ConstantValue::Bool(true), ty),
        ValueType::Index => kernel.index_constant(1),
        ValueType::Bool | ValueType::Vector { .. } | ValueType::Opaque { .. } => {
            panic!("one is undefined for the checked symbolic value type")
        }
    }
}

fn portable_index<B: seismic_target::TargetFamily>(
    kernel: &mut PortableBuilder<'_, B>,
    value: PortableValue,
) -> PortableValue {
    match value.ty() {
        ValueType::Index => value,
        ValueType::Scalar(DType::I32 | DType::U32) => kernel.cast(value, ValueType::Index),
        _ => panic!("checked tensor index has a non-integer portable type"),
    }
}

fn scalar_bits<B: seismic_target::TargetFamily>(
    kernel: &mut PortableBuilder<'_, B>,
    dtype: DType,
    bits: u32,
) -> PortableValue {
    let ty = value_type(dtype);
    let value = match dtype {
        DType::F32 => ConstantValue::F32(f32::from_bits(bits)),
        DType::F16 => ConstantValue::F16(bits as u16),
        DType::BF16 => ConstantValue::BF16(bits as u16),
        DType::I32 => ConstantValue::I32(bits as i32),
        DType::U32 => ConstantValue::U32(bits),
        DType::Bool => ConstantValue::Bool(bits != 0),
    };
    kernel.constant(value, ty)
}

fn lower_scalar_primitive<B: seismic_target::TargetFamily>(
    kernel: &mut PortableBuilder<'_, B>,
    primitive: &PrimitiveId,
    args: &[PortableValue],
    output: &SemanticType,
) -> PortableValue {
    match primitive {
        PrimitiveId::Constant(value) => lower_constant(kernel, *value, output),
        PrimitiveId::Unary(op) => match op {
            ast::UnaryOp::Neg => kernel.unary(UnaryOp::Neg, args[0]),
            ast::UnaryOp::Not => kernel.not(args[0]),
            ast::UnaryOp::BitNot => {
                let ones = match args[0].ty() {
                    ValueType::Scalar(DType::I32) => {
                        kernel.constant(ConstantValue::I32(-1), args[0].ty().clone())
                    }
                    ValueType::Scalar(DType::U32) => {
                        kernel.constant(ConstantValue::U32(u32::MAX), args[0].ty().clone())
                    }
                    _ => panic!("checked bit-not has invalid type"),
                };
                kernel.bit(BitOp::Xor, args[0], ones)
            }
        },
        PrimitiveId::Binary(op) => {
            // Any quantity-to-word conversion is an explicit checked Cast.
            let (lhs, rhs) = (args[0], args[1]);
            match op {
                ast::BinaryOp::Add => kernel.binary(BinaryOp::Add, lhs, rhs),
                ast::BinaryOp::Sub => kernel.binary(BinaryOp::Sub, lhs, rhs),
                ast::BinaryOp::Mul => kernel.binary(BinaryOp::Mul, lhs, rhs),
                ast::BinaryOp::Div => kernel.binary(BinaryOp::Div, lhs, rhs),
                ast::BinaryOp::Rem => kernel.binary(BinaryOp::Rem, lhs, rhs),
                ast::BinaryOp::Eq => kernel.cmp(CmpOp::Eq, lhs, rhs),
                ast::BinaryOp::Ne => kernel.cmp(CmpOp::Ne, lhs, rhs),
                ast::BinaryOp::Lt => kernel.cmp(CmpOp::Lt, lhs, rhs),
                ast::BinaryOp::Le => kernel.cmp(CmpOp::Le, lhs, rhs),
                ast::BinaryOp::Gt => kernel.cmp(CmpOp::Gt, lhs, rhs),
                ast::BinaryOp::Ge => kernel.cmp(CmpOp::Ge, lhs, rhs),
                ast::BinaryOp::And => kernel.logic(LogicOp::And, lhs, rhs),
                ast::BinaryOp::Or => kernel.logic(LogicOp::Or, lhs, rhs),
                ast::BinaryOp::BitAnd => kernel.bit(BitOp::And, lhs, rhs),
                ast::BinaryOp::BitOr => kernel.bit(BitOp::Or, lhs, rhs),
                ast::BinaryOp::BitXor => kernel.bit(BitOp::Xor, lhs, rhs),
                ast::BinaryOp::Shl => kernel.bit(BitOp::Shl, lhs, rhs),
                ast::BinaryOp::Shr => kernel.bit(BitOp::Shr, lhs, rhs),
            }
        }
        PrimitiveId::Cast(dtype) => kernel.cast(
            args[0],
            if matches!(output, SemanticType::Index { .. }) {
                ValueType::Index
            } else {
                value_type(*dtype)
            },
        ),
        PrimitiveId::Math(op) => match op {
            MathOp::Fma => kernel.fma(args[0], args[1], args[2]),
            MathOp::Max => kernel.binary(BinaryOp::Max, args[0], args[1]),
            MathOp::Min => kernel.binary(BinaryOp::Min, args[0], args[1]),
            _ => kernel.math(*op, args[0]),
        },
        PrimitiveId::Select => kernel.select(args[0], args[1], args[2]),
        PrimitiveId::Decode => args[0],
        other => panic!("non-scalar primitive in scalar lowering: {other:?}"),
    }
}

fn lower_constant<B: seismic_target::TargetFamily>(
    kernel: &mut PortableBuilder<'_, B>,
    value: ReferenceScalar,
    output: &SemanticType,
) -> PortableValue {
    match output {
        SemanticType::Scalar(dtype) => {
            assert_eq!(
                *dtype,
                value.dtype(),
                "checked constant type differs from its payload"
            );
            kernel.constant(ConstantValue::from_scalar(value), value_type(*dtype))
        }
        SemanticType::Index { .. } => {
            let value = match value {
                ReferenceScalar::I32(value) => {
                    u64::try_from(value).expect("checked index constant is negative")
                }
                ReferenceScalar::U32(value) => u64::from(value),
                _ => panic!("index constant is not integer"),
            };
            kernel.index_constant(value)
        }
        _ => panic!("constant output is not scalar"),
    }
}

#[derive(Clone)]
enum CapturedExpr {
    Nat(u64),
    Int(i64),
    Bool(bool),
    Scalar(DType, u32),
    NatArgument(NatExpr),
    ScalarArgument(seismic_lang::expr::SymbolId, DType),
    Value(PortableValue),
    Binder(LoopBinderId),
    Unary(ExprUnary, Box<CapturedExpr>),
    Binary(ExprBinary, Box<CapturedExpr>, Box<CapturedExpr>),
    Nary(NaryOp, Vec<CapturedExpr>),
    Select(Box<CapturedExpr>, Box<CapturedExpr>, Box<CapturedExpr>),
    Cmp(
        seismic_lang::expr::CmpOp,
        Box<CapturedExpr>,
        Box<CapturedExpr>,
    ),
    In(Box<CapturedExpr>, Vec<i64>),
    Fold {
        op: FoldOp,
        binder: LoopBinderId,
        start: Box<CapturedExpr>,
        extent: Box<CapturedExpr>,
        body: Box<CapturedExpr>,
    },
}
fn capture_expr(
    arena: &mut ExprArena,
    bindings: &BindingArena,
    expression: AnyExpr,
    values: &SemanticBindings,
) -> CapturedExpr {
    capture_expr_with(arena, expression, &mut |arena, value| {
        let bound = values.get(bindings, value);
        match prepare_scalar(arena, bound.scalar()) {
            PreparedArg::Index(value) => CapturedExpr::NatArgument(value),
            PreparedArg::Integer(_) => panic!("exact Integer requires a reached host operation before native capture"),
            PreparedArg::Scalar(symbol, dtype) => CapturedExpr::ScalarArgument(symbol, dtype),
        }
    })
}

fn capture_expr_with(
    arena: &mut ExprArena,
    expression: AnyExpr,
    runtime: &mut impl FnMut(&mut ExprArena, SemanticValueId) -> CapturedExpr,
) -> CapturedExpr {
    match arena.view(expression) {
        NodeView::NatConst(v) => CapturedExpr::Nat(v),
        NodeView::IntConst(v) => CapturedExpr::Int(v),
        NodeView::BoolConst(v) => CapturedExpr::Bool(v),
        NodeView::ScalarConst { dtype, bits } => CapturedExpr::Scalar(dtype, bits),
        NodeView::ScalarInteger { .. } => {
            panic!("source word projection must resolve to its actual bound result before physical expression construction")
        }
        NodeView::Symbol(symbol) => match arena.symbol_kind(symbol) {
            SymbolKind::RuntimeValue(value) => runtime(arena, value),
            SymbolKind::LoopBinder(binder) => CapturedExpr::Binder(binder),
            _ => match arena.symbol_sort(symbol) {
                SymbolSort::Nat => CapturedExpr::NatArgument(arena.nat_symbol(symbol)),
                SymbolSort::Int => CapturedExpr::ScalarArgument(symbol, DType::I32),
                SymbolSort::Scalar(dtype) => CapturedExpr::ScalarArgument(symbol, dtype),
            },
        },
        NodeView::Unary { op, operand } => {
            CapturedExpr::Unary(op, Box::new(capture_expr_with(arena, operand, runtime)))
        }
        NodeView::Binary { op, lhs, rhs } => CapturedExpr::Binary(
            op,
            Box::new(capture_expr_with(arena, lhs, runtime)),
            Box::new(capture_expr_with(arena, rhs, runtime)),
        ),
        NodeView::Nary { op, operands } => {
            let operands = operands.to_vec();
            CapturedExpr::Nary(
                op,
                operands
                    .into_iter()
                    .map(|value| capture_expr_with(arena, value, runtime))
                    .collect(),
            )
        }
        NodeView::Select {
            cond,
            then,
            otherwise,
        } => CapturedExpr::Select(
            Box::new(capture_expr_with(arena, cond.into(), runtime)),
            Box::new(capture_expr_with(arena, then, runtime)),
            Box::new(capture_expr_with(arena, otherwise, runtime)),
        ),
        NodeView::Cmp { op, lhs, rhs } => CapturedExpr::Cmp(
            op,
            Box::new(capture_expr_with(arena, lhs, runtime)),
            Box::new(capture_expr_with(arena, rhs, runtime)),
        ),
        NodeView::In {
            operand,
            values: members,
        } => {
            let members = members.to_vec();
            CapturedExpr::In(
                Box::new(capture_expr_with(arena, operand, runtime)),
                members,
            )
        }
        NodeView::Fold {
            op,
            binder,
            start,
            extent,
            body,
        } => CapturedExpr::Fold {
            op,
            binder,
            start: Box::new(capture_expr_with(arena, AnyExpr::Nat(start), runtime)),
            extent: Box::new(capture_expr_with(arena, AnyExpr::Nat(extent), runtime)),
            body: Box::new(capture_expr_with(arena, AnyExpr::Nat(body), runtime)),
        },
        NodeView::Duration(_) | NodeView::DurationScale { .. } => {
            unreachable!("typed integer symbolic expression contains duration nodes")
        }
    }
}

/// Select the native natural representation only after the defining source
/// result has proved every evaluated quantity intermediate lies in the Index
/// domain. In particular, an `Int` literal is not inherently an I32 word.
fn native_natural_expr(expression: CapturedExpr) -> Option<CapturedExpr> {
    Some(match expression {
        CapturedExpr::Int(value) => CapturedExpr::Nat(u64::try_from(value).ok()?),
        CapturedExpr::Nat(value) => CapturedExpr::Nat(value),
        CapturedExpr::Bool(value) => CapturedExpr::Bool(value),
        CapturedExpr::NatArgument(value) => CapturedExpr::NatArgument(value),
        CapturedExpr::Binder(value) => CapturedExpr::Binder(value),
        CapturedExpr::Value(value) if value.ty() == ValueType::Index => {
            CapturedExpr::Value(value)
        }
        CapturedExpr::Unary(ExprUnary::IntFromNat | ExprUnary::NatFromInt, value) => {
            native_natural_expr(*value)?
        }
        CapturedExpr::Unary(op, value) => {
            CapturedExpr::Unary(op, Box::new(native_natural_expr(*value)?))
        }
        CapturedExpr::Binary(op, a, b) => CapturedExpr::Binary(
            op,
            Box::new(native_natural_expr(*a)?),
            Box::new(native_natural_expr(*b)?),
        ),
        CapturedExpr::Nary(op, values) => CapturedExpr::Nary(
            op,
            values.into_iter().map(native_natural_expr).collect::<Option<Vec<_>>>()?,
        ),
        CapturedExpr::Select(condition, yes, no) => CapturedExpr::Select(
            Box::new(native_natural_expr(*condition)?),
            Box::new(native_natural_expr(*yes)?),
            Box::new(native_natural_expr(*no)?),
        ),
        CapturedExpr::Cmp(op, a, b) => CapturedExpr::Cmp(
            op,
            Box::new(native_natural_expr(*a)?),
            Box::new(native_natural_expr(*b)?),
        ),
        CapturedExpr::In(value, members) => CapturedExpr::In(
            Box::new(native_natural_expr(*value)?),
            members.into_iter().filter(|value| *value >= 0).collect(),
        ),
        CapturedExpr::Scalar(..)
        | CapturedExpr::ScalarArgument(..)
        | CapturedExpr::Value(_)
        | CapturedExpr::Fold { .. } => return None,
    })
}

/// A prepared tensor lacks the source participant envelope at this point.
/// Only an already native natural input can be consumed without proving new
/// arithmetic. Participant-defined results use `shape_native_natural` first.
fn direct_native_natural_expr(expression: &CapturedExpr) -> bool {
    match expression {
        CapturedExpr::Nat(_) | CapturedExpr::NatArgument(_) | CapturedExpr::Binder(_) => true,
        CapturedExpr::Value(value) => value.ty() == ValueType::Index,
        _ => false,
    }
}
fn captured_uniformity<B: seismic_target::TargetFamily>(
    kernel: &PortableBuilder<'_, B>,
    expression: &CapturedExpr,
    binders: &BTreeMap<LoopBinderId, registry::IntrinsicUniformity>,
) -> registry::IntrinsicUniformity {
    use registry::IntrinsicUniformity as U;
    match expression {
        CapturedExpr::Nat(_)
        | CapturedExpr::Int(_)
        | CapturedExpr::Bool(_)
        | CapturedExpr::Scalar(..)
        | CapturedExpr::NatArgument(_)
        | CapturedExpr::ScalarArgument(..) => U::Workgroup,
        CapturedExpr::Value(value) => kernel.uniformity(*value),
        CapturedExpr::Binder(binder) => *binders
            .get(binder)
            .expect("symbolic fold uniformity requires its lexical binder"),
        CapturedExpr::Unary(_, value) | CapturedExpr::In(value, _) => {
            captured_uniformity(kernel, value, binders)
        }
        CapturedExpr::Binary(_, a, b) | CapturedExpr::Cmp(_, a, b) => uniformity::join(
            captured_uniformity(kernel, a, binders),
            captured_uniformity(kernel, b, binders),
        ),
        CapturedExpr::Nary(_, values) => uniformity::all(
            values
                .iter()
                .map(|value| captured_uniformity(kernel, value, binders)),
        ),
        CapturedExpr::Select(condition, a, b) => uniformity::all(
            [condition, a, b]
                .into_iter()
                .map(|value| captured_uniformity(kernel, value, binders)),
        ),
        CapturedExpr::Fold {
            binder,
            start,
            extent,
            body,
            ..
        } => {
            let range = uniformity::join(
                captured_uniformity(kernel, start, binders),
                captured_uniformity(kernel, extent, binders),
            );
            let mut nested = binders.clone();
            nested.insert(*binder, range);
            uniformity::join(range, captured_uniformity(kernel, body, &nested))
        }
    }
}
fn lower_captured_expr<B: seismic_target::TargetFamily>(
    kernel: &mut PortableBuilder<'_, B>,
    expression: &CapturedExpr,
) -> PortableValue {
    lower_captured_expr_with(kernel, expression, &mut BTreeMap::new())
}
fn lower_captured_expr_with<B: seismic_target::TargetFamily>(
    kernel: &mut PortableBuilder<'_, B>,
    expression: &CapturedExpr,
    binders: &mut BTreeMap<LoopBinderId, PortableValue>,
) -> PortableValue {
    match expression {
        CapturedExpr::Nat(v) => kernel.index_constant(*v),
        CapturedExpr::Int(v) => kernel.constant(
            ConstantValue::I32(i32::try_from(*v).expect("checked symbolic integer exceeds i32")),
            ValueType::Scalar(DType::I32),
        ),
        CapturedExpr::Bool(v) => kernel.constant(ConstantValue::Bool(*v), ValueType::Bool),
        CapturedExpr::Scalar(dtype, bits) => scalar_bits(kernel, *dtype, *bits),
        CapturedExpr::NatArgument(expression) => kernel.nat_arg(*expression),
        CapturedExpr::ScalarArgument(symbol, dtype) => kernel.scalar_arg(*symbol, *dtype),
        CapturedExpr::Value(value) => *value,
        CapturedExpr::Binder(binder) => *binders
            .get(binder)
            .expect("symbolic fold body references a binder outside its lexical fold"),
        CapturedExpr::Unary(op, a) => {
            let a = lower_captured_expr_with(kernel, a, binders);
            match op {
                ExprUnary::Not => kernel.not(a),
                ExprUnary::NatFromInt => kernel.cast(a, ValueType::Index),
                ExprUnary::IntFromNat => kernel.cast(a, ValueType::Scalar(DType::I32)),
                // Injection preserves the actual word representation. The
                // mathematical operation that consumes it owns its width.
                ExprUnary::IntFromScalar => a,
                ExprUnary::ScalarIntegerDefined => {
                    panic!("source recipe definedness belongs to its source continuation")
                }
            }
        }
        CapturedExpr::Binary(op, a, b) => {
            let a = lower_captured_expr_with(kernel, a, binders);
            let b = lower_captured_expr_with(kernel, b, binders);
            match op {
                ExprBinary::Add => kernel.binary(BinaryOp::Add, a, b),
                ExprBinary::Sub => kernel.binary(BinaryOp::Sub, a, b),
                ExprBinary::Mul => kernel.binary(BinaryOp::Mul, a, b),
                ExprBinary::Div => kernel.binary(BinaryOp::Div, a, b),
                ExprBinary::Rem => kernel.binary(BinaryOp::Rem, a, b),
                ExprBinary::Min => kernel.binary(BinaryOp::Min, a, b),
                ExprBinary::Max => kernel.binary(BinaryOp::Max, a, b),
                ExprBinary::CeilDiv => {
                    let q = kernel.binary(BinaryOp::Div, a, b);
                    let r = kernel.binary(BinaryOp::Rem, a, b);
                    let zero = zero_of(kernel, r.ty().clone());
                    let more = kernel.cmp(CmpOp::Gt, r, zero);
                    let one = one_of(kernel, q.ty().clone());
                    let zero = zero_of(kernel, q.ty().clone());
                    let increment = kernel.select(more, one, zero);
                    kernel.binary(BinaryOp::Add, q, increment)
                }
                ExprBinary::AlignUp => {
                    let q = kernel.binary(BinaryOp::Div, a, b);
                    let r = kernel.binary(BinaryOp::Rem, a, b);
                    let zero = zero_of(kernel, r.ty().clone());
                    let more = kernel.cmp(CmpOp::Gt, r, zero);
                    let one = one_of(kernel, q.ty().clone());
                    let zero = zero_of(kernel, q.ty().clone());
                    let increment = kernel.select(more, one, zero);
                    let q = kernel.binary(BinaryOp::Add, q, increment);
                    kernel.binary(BinaryOp::Mul, q, b)
                }
                ExprBinary::And => kernel.logic(LogicOp::And, a, b),
                ExprBinary::Or => kernel.logic(LogicOp::Or, a, b),
                ExprBinary::Implies => {
                    let not_a = kernel.not(a);
                    kernel.logic(LogicOp::Or, not_a, b)
                }
                ExprBinary::Iff => kernel.cmp(CmpOp::Eq, a, b),
            }
        }
        CapturedExpr::Nary(op, items) => {
            let mut lowered = Vec::with_capacity(items.len());
            for item in items {
                lowered.push(lower_captured_expr_with(kernel, item, binders));
            }
            let mut lowered = lowered.into_iter();
            let mut result = lowered.next().expect("empty symbolic nary");
            for value in lowered {
                result = match op {
                    NaryOp::Product => kernel.binary(BinaryOp::Mul, result, value),
                    NaryOp::All => kernel.logic(LogicOp::And, result, value),
                    NaryOp::Any => kernel.logic(LogicOp::Or, result, value),
                    NaryOp::DurationAdd => {
                        unreachable!("integer symbolic expression contains duration sum")
                    }
                };
            }
            result
        }
        CapturedExpr::Select(c, a, b) => {
            let c = lower_captured_expr_with(kernel, c, binders);
            let a = lower_captured_expr_with(kernel, a, binders);
            let b = lower_captured_expr_with(kernel, b, binders);
            kernel.select(c, a, b)
        }
        CapturedExpr::Cmp(op, a, b) => {
            let a = lower_captured_expr_with(kernel, a, binders);
            let b = lower_captured_expr_with(kernel, b, binders);
            let op = match op {
                seismic_lang::expr::CmpOp::Eq => CmpOp::Eq,
                seismic_lang::expr::CmpOp::Ne => CmpOp::Ne,
                seismic_lang::expr::CmpOp::Lt => CmpOp::Lt,
                seismic_lang::expr::CmpOp::Le => CmpOp::Le,
                seismic_lang::expr::CmpOp::Gt => CmpOp::Gt,
                seismic_lang::expr::CmpOp::Ge => CmpOp::Ge,
            };
            kernel.cmp(op, a, b)
        }
        CapturedExpr::In(operand, values) => {
            let operand = lower_captured_expr_with(kernel, operand, binders);
            let mut result = kernel.constant(ConstantValue::Bool(false), ValueType::Bool);
            for value in values {
                let constant = match operand.ty() {
                    ValueType::Index => kernel.index_constant(
                        u64::try_from(*value).expect("Nat membership contains a negative value"),
                    ),
                    ValueType::Scalar(DType::I32) => kernel.constant(
                        ConstantValue::I32(
                            i32::try_from(*value).expect("membership value exceeds i32"),
                        ),
                        ValueType::Scalar(DType::I32),
                    ),
                    _ => unreachable!("membership operand has a noninteger type"),
                };
                let equal = kernel.cmp(CmpOp::Eq, operand, constant);
                result = kernel.logic(LogicOp::Or, result, equal);
            }
            result
        }
        CapturedExpr::Fold {
            op,
            binder,
            start,
            extent,
            body,
        } => {
            let start = lower_captured_expr_with(kernel, start, binders);
            let extent = lower_captured_expr_with(kernel, extent, binders);
            let end = kernel.binary(BinaryOp::Add, start, extent);
            let initial = match op {
                FoldOp::Sum | FoldOp::Max => kernel.index_constant(0),
                FoldOp::Product => kernel.index_constant(1),
            };
            let mut scopes = binders
                .iter()
                .map(|(binder, value)| (*binder, kernel.uniformity(*value)))
                .collect::<BTreeMap<_, _>>();
            scopes.insert(
                *binder,
                uniformity::join(kernel.uniformity(start), kernel.uniformity(end)),
            );
            let recurrence = captured_uniformity(kernel, body, &scopes);
            let result = kernel.repeat(
                start,
                end,
                vec![initial],
                &[recurrence],
                |kernel, value, carry| {
                    binders.insert(*binder, value);
                    let term = lower_captured_expr_with(kernel, body, binders);
                    binders.remove(binder);
                    let result = match op {
                        FoldOp::Sum => kernel.binary(BinaryOp::Add, carry[0], term),
                        FoldOp::Product => kernel.binary(BinaryOp::Mul, carry[0], term),
                        FoldOp::Max => kernel.binary(BinaryOp::Max, carry[0], term),
                    };
                    vec![result]
                },
            );
            result[0]
        }
    }
}

#[cfg(test)]
mod import_tests;
