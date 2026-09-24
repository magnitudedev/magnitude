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
mod bindings;
mod quantity;
mod calls;
mod regions;
use bindings::{
    Bound, LoopConstruction, SegmentBound, SegmentTensor, SemanticBindings, TensorDefinition,
    TensorDefinitionValue, TensorRealization,
};
pub use bindings::BindingId;
pub(crate) use bindings::{
    initialization_context, BindingArena, BindingPath, BindingPhysicalImport, BindingSelections,
    BindingSelector, FrozenBindings,
};
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
    // A stream plan is a schedule-level value: its actual axes are host
    // expressions at the launch that instantiates it.
    let axes = plan.axes.iter().map(|axis| kernel.nat_arg(*axis)).collect();
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
        let SemanticType::Tensor(_) = &self.function.value(value).ty else {
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
                // An axis broadcast with itself is itself.
                if *target == *source { continue; }
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
                registry::IntrinsicParticipation::FullWorkgroup
                | registry::IntrinsicParticipation::FixedWorkgroup(_) => Some(participants),
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
        RepresentationKind::Packed(_) | RepresentationKind::PackedRows(_) => DType::F32,
        RepresentationKind::External(_) => {
            panic!("external artifact representation has no element-read dtype")
        }
    }
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

/// A Boolean value as a host predicate: its exact defining comparison when a
/// host evaluation of exact quantities defined it, otherwise its slot.
fn condition_expr(
    arena: &mut ExprArena,
    host_conditions: &std::collections::HashMap<seismic_lang::expr::SymbolId, seismic_lang::expr::BoolExpr>,
    bound: &Bound,
) -> seismic_lang::expr::BoolExpr {
    let PreparedArg::Scalar(symbol, DType::Bool) = prepare_scalar(arena, bound.scalar()) else {
        panic!("checked condition is not Boolean")
    };
    if let Some(condition) = host_conditions.get(&symbol) {
        return *condition;
    }
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
