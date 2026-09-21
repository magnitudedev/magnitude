//! Canonical lowering of checked portable semantics into a closed implementation.
//!
//! This is deliberately a direct constructor, not a recipe or fallback.  It
//! emits a sequential, one-participant schedule whose only choices are the
//! source program's structured control. Backend factories are optimized peers.

use crate::implementation::{
    Applicability, FactoryIdentity, FactoryRequest, ImplementationBuilder, ImplementationDraft,
    ImplementationFactory, ScalarPublication, ValueBinding,
};
use crate::kernel::internals::{PortableBuilder, PortableTensor, PortableValue};
use crate::kernel::ops::{
    BinaryOp, BitOp, CheckSite, CmpOp, ConstantValue, LogicOp, SegmentLaunchDomain,
    SemanticIntrinsicCall, SemanticIntrinsicOperand, SemanticIntrinsicResult,
    SemanticIntrinsicSink, SemanticPlace, SemanticScalar, UnaryOp, ValueType,
};
use crate::schedule::{AnyScalarSlot, LaunchMode};
use crate::storage::{AnyBufferView, LaunchLocalKind};
use crate::target::Backend;
use seismic_lang::entry::{
    CheckReason, LoopKind, RegionKind, ScalarRef, SemanticFunction, SemanticNodeView, SemanticType,
    SliceAxis, TensorSemantics, TensorStorage, ValueOrigin, ViewTransform,
};
use seismic_lang::expr::{
    AnyExpr, BinaryOp as ExprBinary, ExprArena, FiniteDomain, FoldOp, LoopBinderId, NaryOp,
    NatExpr, NodeView, SymbolKind, SymbolSort, UnaryOp as ExprUnary,
};
use seismic_lang::ids::{FamilyId, FunctionId, IntrinsicId, NodeId, RegionId, SemanticValueId};
use seismic_lang::intrinsics::{reduce_schema, Constant, MathOp, PrimitiveId, ReduceOp};
use seismic_lang::registry::{self, RepresentationKind};
use seismic_lang::syntax::ast;
use seismic_lang::types::DType;
use std::collections::{BTreeMap, BTreeSet};

pub(crate) struct PortableFactory;
pub(crate) struct PortableParallelFactory;
pub(crate) struct AuthoredSemanticFactory;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SemanticMode {
    Portable,
    PortableParallel,
    AuthoredBackend,
}

impl<B: Backend> ImplementationFactory<B> for PortableFactory {
    fn identity(&self) -> FactoryIdentity {
        FactoryIdentity {
            name: "portable.semantic",
            revision: "1",
        }
    }

    fn applicable(&self, request: &FactoryRequest<'_, B>) -> Applicability {
        if matches!(
            request.candidate_kind,
            seismic_lang::entry::CandidateKind::Portable
        ) {
            Applicability::Applicable
        } else {
            Applicability::NotApplicable {
                reason: "portable semantic factory only constructs portable candidates".into(),
            }
        }
    }

    fn construct(
        &self,
        request: &FactoryRequest<'_, B>,
        builder: ImplementationBuilder<'_, B>,
    ) -> ImplementationDraft<B> {
        construct_semantic(request.function, builder, SemanticMode::Portable)
    }
}

impl<B: Backend> ImplementationFactory<B> for PortableParallelFactory {
    fn identity(&self) -> FactoryIdentity {
        FactoryIdentity {
            name: "portable.parallel",
            revision: "1",
        }
    }

    fn applicable(&self, request: &FactoryRequest<'_, B>) -> Applicability {
        if matches!(
            request.candidate_kind,
            seismic_lang::entry::CandidateKind::Portable
        ) {
            Applicability::Applicable
        } else {
            Applicability::NotApplicable {
                reason: "parallel portable factory only constructs portable candidates".into(),
            }
        }
    }

    fn construct(
        &self,
        request: &FactoryRequest<'_, B>,
        builder: ImplementationBuilder<'_, B>,
    ) -> ImplementationDraft<B> {
        construct_semantic(request.function, builder, SemanticMode::PortableParallel)
    }
}

impl<B: Backend> ImplementationFactory<B> for AuthoredSemanticFactory {
    fn identity(&self) -> FactoryIdentity {
        FactoryIdentity {
            name: "authored.semantic",
            revision: "1",
        }
    }

    fn applicable(&self, request: &FactoryRequest<'_, B>) -> Applicability {
        match request.candidate_kind {
            seismic_lang::entry::CandidateKind::Lowering { backend }
            | seismic_lang::entry::CandidateKind::Helper { backend }
                if backend == B::NAME =>
            {
                Applicability::Applicable
            }
            _ => Applicability::NotApplicable {
                reason: "authored semantic factory requires a candidate for this backend".into(),
            },
        }
    }

    fn construct(
        &self,
        request: &FactoryRequest<'_, B>,
        builder: ImplementationBuilder<'_, B>,
    ) -> ImplementationDraft<B> {
        construct_semantic(request.function, builder, SemanticMode::AuthoredBackend)
    }
}

fn construct_semantic<'a, B: Backend>(
    function: &'a SemanticFunction,
    mut builder: ImplementationBuilder<'a, B>,
    mode: SemanticMode,
) -> ImplementationDraft<B> {
    let mut lowerer = Lowerer::new(function, &mut builder, mode);
    lowerer.lower_region(function.root());
    lowerer.publish_results();
    let schedule = builder.schedule().close();
    builder.close(schedule)
}

/// Returns the launch-global logical coordinate. Universal portable kernels
/// receive a compiler-owned base argument so later schedule specialization can
/// split one semantic launch without changing the kernel's meaning.
fn logical_global_id<B: Backend>(
    kernel: &mut PortableBuilder<'_, B>,
    compiler_owned: bool,
    zero: NatExpr,
) -> (PortableValue, Option<u32>) {
    let physical = kernel.global_id(0);
    if compiler_owned {
        let (base, argument) = kernel.nat_arg_with_ordinal(zero);
        (kernel.binary(BinaryOp::Add, base, physical), Some(argument))
    } else {
        (physical, None)
    }
}

#[derive(Clone, Debug)]
enum Bound {
    Tensor(AnyBufferView),
    Scalar {
        symbol: seismic_lang::expr::SymbolId,
        dtype: DType,
        index: bool,
        direct: Option<NatExpr>,
        slot: Option<AnyScalarSlot>,
    },
    Range {
        start: Box<Bound>,
        end: Box<Bound>,
    },
    Tuple(Vec<Bound>),
    Unit,
}

#[derive(Clone, Debug)]
enum SegmentBound {
    Tensor(SegmentTensor),
    Scalar(PortableValue),
    Opaque(crate::kernel::ops::SemanticOpaque),
    Range {
        start: Box<SegmentBound>,
        end: Box<SegmentBound>,
    },
    Tuple(Vec<SegmentBound>),
    Unit,
}

#[derive(Clone, Debug)]
struct SegmentTensor {
    axes: Vec<PortableValue>,
    value: SegmentTensorValue,
}

#[derive(Clone, Debug)]
enum SegmentTensorValue {
    Physical(PortableTensor),
    Elementwise {
        primitive: PrimitiveId,
        inputs: Vec<SegmentBound>,
        input_axes: Vec<Option<Vec<PortableValue>>>,
        output: SemanticType,
    },
    Reduce {
        op: ReduceOp,
        axis: usize,
        input_dtype: DType,
        input: Box<SegmentTensor>,
    },
    View {
        base: Box<SegmentTensor>,
        transform: SegmentViewTransform,
    },
}

#[derive(Clone, Debug)]
enum SegmentViewTransform {
    Slice(Vec<SegmentSliceAxis>),
    Transpose(Vec<u32>),
    Reshape,
}

#[derive(Clone, Debug)]
enum SegmentSliceAxis {
    Full,
    Point(PortableValue),
    Range { start: PortableValue },
}

impl SegmentTensor {
    fn physical<B: Backend>(kernel: &PortableBuilder<'_, B>, value: PortableTensor) -> Self {
        Self {
            axes: kernel.tensor_extents(&value).to_vec(),
            value: SegmentTensorValue::Physical(value),
        }
    }

    fn read<B: Backend>(
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
            SegmentTensorValue::Physical(value) => kernel.tensor_read(value, index),
            SegmentTensorValue::Elementwise {
                primitive,
                inputs,
                input_axes,
                output,
            } => {
                let mut arguments = Vec::with_capacity(inputs.len());
                for (input, input_axes) in inputs.iter().zip(input_axes) {
                    arguments.push(match input {
                        SegmentBound::Scalar(value) => *value,
                        SegmentBound::Tensor(tensor) => {
                            let skip = index.len() - tensor.axes.len();
                            let axes = input_axes
                                .as_ref()
                                .expect("tensor expression input has no axes");
                            let one = kernel.index_constant(1);
                            let zero = kernel.index_constant(0);
                            let source = index[skip..]
                                .iter()
                                .zip(axes)
                                .map(|(value, extent)| {
                                    let broadcast = kernel.cmp(CmpOp::Eq, *extent, one);
                                    kernel.select(broadcast, zero, *value)
                                })
                                .collect::<Vec<_>>();
                            tensor.read(kernel, &source)
                        }
                        _ => panic!("elementwise expression input is not scalar or tensor"),
                    });
                }
                lower_scalar_primitive(kernel, primitive, &arguments, output)
            }
            SegmentTensorValue::Reduce {
                op,
                axis,
                input_dtype,
                input,
            } => {
                let schema = reduce_schema(*op, *input_dtype);
                let zero = kernel.index_constant(0);
                let end = input.axes[*axis];
                let first_index = reduction_index(index, *axis, zero);
                let first = input.read(kernel, &first_index);
                let first = if first.ty == value_type(schema.accumulator) {
                    first
                } else {
                    kernel.cast(first, value_type(schema.accumulator))
                };
                let initial = match op {
                    ReduceOp::Sum => zero_of(kernel, first.ty.clone()),
                    ReduceOp::Max | ReduceOp::Min | ReduceOp::Argmax => first,
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
                let result = kernel.repeat(start, end, carries, |kernel, binder, carry| {
                    let source_index = reduction_index(index, *axis, binder);
                    let value = input.read(kernel, &source_index);
                    let value = if value.ty == value_type(schema.accumulator) {
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
            SegmentTensorValue::View { base, transform } => {
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
                        assert!(values.next().is_none(), "slice mapping left an output axis");
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
        view: AnyBufferView,
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

#[derive(Clone, Debug)]
struct StreamTensorPlan {
    axes: Vec<NatExpr>,
    value: StreamTensorPlanValue,
}

#[derive(Clone, Debug)]
enum StreamTensorPlanValue {
    Physical(AnyBufferView),
    Elementwise {
        primitive: PrimitiveId,
        inputs: Vec<StreamBoundPlan>,
        input_axes: Vec<Option<Vec<NatExpr>>>,
        output: SemanticType,
    },
    View {
        base: Box<StreamTensorPlan>,
        transform: StreamViewPlan,
    },
}

#[derive(Clone, Debug)]
enum StreamBoundPlan {
    Tensor(StreamTensorPlan),
    Scalar(PreparedArg),
}

#[derive(Clone, Debug)]
enum StreamViewPlan {
    Slice(Vec<StreamSliceAxisPlan>),
    Transpose(Vec<u32>),
    Reshape,
}

#[derive(Clone, Debug)]
enum StreamSliceAxisPlan {
    Full,
    Point(PreparedArg),
    Range { start: PreparedArg },
}

fn instantiate_stream_tensor<B: Backend>(
    kernel: &mut PortableBuilder<'_, B>,
    plan: &StreamTensorPlan,
) -> SegmentTensor {
    let axes = plan
        .axes
        .iter()
        .map(|axis| kernel.nat_arg(*axis))
        .collect::<Vec<_>>();
    let value = match &plan.value {
        StreamTensorPlanValue::Physical(view) => {
            let place = kernel.arg_view(*view, false);
            SegmentTensorValue::Physical(kernel.tensor(place))
        }
        StreamTensorPlanValue::Elementwise {
            primitive,
            inputs,
            input_axes,
            output,
        } => SegmentTensorValue::Elementwise {
            primitive: primitive.clone(),
            inputs: inputs
                .iter()
                .map(|input| match input {
                    StreamBoundPlan::Tensor(tensor) => {
                        SegmentBound::Tensor(instantiate_stream_tensor(kernel, tensor))
                    }
                    StreamBoundPlan::Scalar(value) => {
                        SegmentBound::Scalar(prepared_kernel_arg(kernel, *value))
                    }
                })
                .collect(),
            input_axes: input_axes
                .iter()
                .map(|axes| {
                    axes.as_ref().map(|axes| {
                        axes.iter()
                            .map(|axis| kernel.nat_arg(*axis))
                            .collect::<Vec<_>>()
                    })
                })
                .collect(),
            output: output.clone(),
        },
        StreamTensorPlanValue::View { base, transform } => {
            let base = Box::new(instantiate_stream_tensor(kernel, base));
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
            SegmentTensorValue::View { base, transform }
        }
    };
    SegmentTensor { axes, value }
}

fn prepared_kernel_arg<B: Backend>(
    kernel: &mut PortableBuilder<'_, B>,
    argument: PreparedArg,
) -> PortableValue {
    match argument {
        PreparedArg::Index(value) => kernel.nat_arg(value),
        PreparedArg::Scalar(symbol, dtype) => kernel.scalar_arg(symbol, dtype),
    }
}

/// Total construction environment for one checked function. Owner and
/// ordinal checks are concentrated here; lowering sites never join raw IDs
/// against an unqualified map.
#[derive(Clone, Debug)]
struct SemanticBindings {
    function: FunctionId,
    values: Vec<Option<Bound>>,
}

impl SemanticBindings {
    fn new(function: &SemanticFunction) -> Self {
        Self {
            function: function.id(),
            values: vec![None; function.values().count()],
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

    fn get(&self, value: SemanticValueId) -> Bound {
        self.values[self.slot(value)]
            .clone()
            .expect("checked semantic value was used before its dominating definition")
    }

    fn bind(&mut self, value: SemanticValueId, bound: Bound) {
        let slot = self.slot(value);
        assert!(
            self.values[slot].replace(bound).is_none(),
            "semantic value was bound twice"
        );
    }

    fn rebind(&mut self, value: SemanticValueId, bound: Bound) {
        let slot = self.slot(value);
        self.values[slot] = Some(bound);
    }

    fn contains(&self, value: SemanticValueId) -> bool {
        self.values[self.slot(value)].is_some()
    }
}

impl Bound {
    fn scalar(&self) -> (seismic_lang::expr::SymbolId, DType, bool) {
        match self {
            Self::Scalar {
                symbol,
                dtype,
                index,
                ..
            } => (*symbol, *dtype, *index),
            _ => panic!("checked scalar value has no scalar portable binding"),
        }
    }
    fn tensor(&self) -> AnyBufferView {
        match self {
            Self::Tensor(view) => *view,
            _ => panic!("checked tensor value has no tensor binding"),
        }
    }
}

struct Lowerer<'f, 'b, B: Backend> {
    function: &'f SemanticFunction,
    builder: &'b mut ImplementationBuilder<'f, B>,
    values: SemanticBindings,
    mode: SemanticMode,
    streamed_values: BTreeSet<SemanticValueId>,
}

impl<'f, 'b, B: Backend> Lowerer<'f, 'b, B> {
    fn new(
        function: &'f SemanticFunction,
        builder: &'b mut ImplementationBuilder<'f, B>,
        mode: SemanticMode,
    ) -> Self {
        let mut values = SemanticBindings::new(function);
        for parameter in function.parameters() {
            let bound = match builder.portable_binding(parameter.value) {
                ValueBinding::View { view, .. } => Bound::Tensor(view),
                ValueBinding::Scalar(symbol) => match function.value(parameter.value).ty {
                    SemanticType::Scalar(dtype) => Bound::Scalar {
                        symbol,
                        dtype,
                        index: false,
                        direct: None,
                        slot: None,
                    },
                    SemanticType::Index { .. } => Bound::Scalar {
                        symbol,
                        dtype: DType::U32,
                        index: true,
                        direct: None,
                        slot: None,
                    },
                    _ => panic!("checked scalar parameter binding has non-scalar type"),
                },
                ValueBinding::Range { start, end } => Bound::Range {
                    start: Box::new(Bound::Scalar {
                        symbol: start,
                        dtype: DType::U32,
                        index: true,
                        direct: None,
                        slot: None,
                    }),
                    end: Box::new(Bound::Scalar {
                        symbol: end,
                        dtype: DType::U32,
                        index: true,
                        direct: None,
                        slot: None,
                    }),
                },
            };
            values.bind(parameter.value, bound);
        }
        Self {
            function,
            builder,
            values,
            mode,
            // Stream fusion is an optional portable optimization. Keep the
            // ordinary materialized path as the construction default; sites
            // that prove a closed streamed subgraph can populate this set.
            streamed_values: BTreeSet::new(),
        }
    }

    fn lower_region(&mut self, region: RegionId) {
        let nodes = self
            .function
            .nodes(region)
            .map(|(id, _)| id)
            .collect::<Vec<_>>();
        for node in nodes {
            self.lower_node(node);
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
        let adjusted = self.builder.arena().nat_add(extent, workgroup);
        let adjusted = self.builder.arena().nat_sub(adjusted, one);
        let groups = self.builder.arena().nat_div(adjusted, workgroup);
        let empty = self
            .builder
            .arena()
            .nat_cmp(seismic_lang::expr::CmpOp::Eq, extent, zero);
        SegmentLaunchDomain {
            mode: LaunchMode::Independent,
            grid: [groups, one, one],
            workgroup: [workgroup, one, one],
            empty,
            parallel_extent: extent,
        }
    }

    /// Realizes exactly one pure semantic dependency DAG. This is used by
    /// optimized factories that request a checked tensor view without walking
    /// (and therefore without replaying) the function's effectful root prefix.
    fn realize_pure_value(
        &mut self,
        value: SemanticValueId,
        active: &mut BTreeSet<SemanticValueId>,
    ) {
        if self.values.contains(value) {
            return;
        }
        if let Some(binding) = self.builder.portable_existing_binding(value) {
            let bound = match binding {
                ValueBinding::View { view, .. } => Bound::Tensor(view),
                ValueBinding::Scalar(symbol) => match self.function.value(value).ty {
                    SemanticType::Scalar(dtype) => Bound::Scalar {
                        symbol,
                        dtype,
                        index: false,
                        direct: None,
                        slot: None,
                    },
                    SemanticType::Index { .. } => Bound::Scalar {
                        symbol,
                        dtype: DType::U32,
                        index: true,
                        direct: None,
                        slot: None,
                    },
                    _ => panic!("existing scalar binding has non-scalar semantic type"),
                },
                ValueBinding::Range { start, end } => Bound::Range {
                    start: Box::new(Bound::Scalar {
                        symbol: start,
                        dtype: DType::U32,
                        index: true,
                        direct: None,
                        slot: None,
                    }),
                    end: Box::new(Bound::Scalar {
                        symbol: end,
                        dtype: DType::U32,
                        index: true,
                        direct: None,
                        slot: None,
                    }),
                },
            };
            self.values.bind(value, bound);
            return;
        }
        if let SemanticType::Tensor(tensor) = &self.function.value(value).ty {
            match &tensor.storage {
                TensorStorage::Parameter(_) => {
                    panic!("checked parameter tensor was not bound to its call argument")
                }
                TensorStorage::Owned => {
                    let view = self.builder.portable_allocate_tensor(value);
                    self.values.bind(value, Bound::Tensor(view));
                    return;
                }
                TensorStorage::View { .. } | TensorStorage::Computed => {}
            }
        }
        assert!(
            active.insert(value),
            "checked pure semantic dependency graph contains a cycle"
        );
        let semantic = self.function.value(value);
        let node = match semantic.origin {
            ValueOrigin::Parameter => {
                panic!("checked parameter tensor was not bound to its call argument")
            }
            ValueOrigin::RegionParameter(_) => {
                panic!("region-local value escaped its checked lexical region")
            }
            ValueOrigin::Node(node) => node,
        };
        let producer = self.function.node(node);
        match producer.view() {
            SemanticNodeView::Primitive { .. }
            | SemanticNodeView::Elementwise { .. }
            | SemanticNodeView::Reduce { .. }
            | SemanticNodeView::View { .. }
            | SemanticNodeView::ElementRead { .. }
            | SemanticNodeView::TuplePack { .. }
            | SemanticNodeView::TupleGet { .. }
            | SemanticNodeView::Extent { .. } => {}
            SemanticNodeView::Intrinsic { .. }
            | SemanticNodeView::Call { .. }
            | SemanticNodeView::Alloc { .. }
            | SemanticNodeView::Fill { .. }
            | SemanticNodeView::Copy { .. }
            | SemanticNodeView::RepresentationConvert { .. }
            | SemanticNodeView::ElementWrite { .. }
            | SemanticNodeView::Store { .. }
            | SemanticNodeView::Atomic { .. }
            | SemanticNodeView::If { .. }
            | SemanticNodeView::Loop { .. }
            | SemanticNodeView::Check { .. } => {
                panic!("effectful semantic value was not realized by its owning traversal")
            }
        }
        for dependency in producer.dependencies() {
            self.realize_pure_value(dependency, active);
        }
        self.lower_node(node);
        active.remove(&value);
        assert!(
            self.values.contains(value),
            "pure semantic producer did not bind its requested value"
        );
    }

    fn lower_node(&mut self, id: NodeId) {
        let node = self.function.node(id);
        let written_places = node
            .events()
            .iter()
            .filter_map(|event| match event.access() {
                seismic_lang::entry::AccessKind::Write(_)
                | seismic_lang::entry::AccessKind::AtomicRmw { .. } => event.place(),
                seismic_lang::entry::AccessKind::Read
                | seismic_lang::entry::AccessKind::Barrier(_) => None,
            })
            .collect::<Vec<_>>();
        match node.view() {
            SemanticNodeView::Primitive {
                primitive,
                inputs,
                output,
            } => self.lower_primitive(primitive, inputs, output),
            SemanticNodeView::Intrinsic {
                intrinsic,
                inputs,
                output,
            } => self.lower_intrinsic(intrinsic, inputs, output),
            SemanticNodeView::Elementwise {
                primitive,
                inputs,
                output,
            } => self.lower_elementwise(primitive, inputs, output),
            SemanticNodeView::Reduce {
                op,
                axis,
                input,
                output,
                ..
            } => self.lower_reduce(op, axis, input, output),
            SemanticNodeView::Call {
                inputs, outputs, ..
            } => self.lower_call(id, inputs, outputs),
            SemanticNodeView::Alloc { output } => {
                let view = self.builder.portable_allocate_tensor(output);
                self.values.bind(output, Bound::Tensor(view));
            }
            SemanticNodeView::Fill { value, output } => {
                let view = self.builder.portable_allocate_tensor(output);
                self.builder.schedule().fill_constant_any(view, value);
                self.values.bind(output, Bound::Tensor(view));
            }
            SemanticNodeView::Copy { input, output } => self.lower_copy(input, output),
            SemanticNodeView::RepresentationConvert { .. } => self.lower_representation_convert(id),
            SemanticNodeView::View {
                base,
                transform,
                output,
            } => self.lower_view(base, transform, output),
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
            SemanticNodeView::If {
                condition,
                captures,
                outputs,
                then,
                otherwise,
            } => self.lower_if(condition, captures, outputs, then, otherwise),
            SemanticNodeView::Loop {
                kind,
                start,
                end,
                captures,
                body,
                carries,
                ..
            } => self.lower_loop(kind, start, end, captures, body, carries, &written_places),
            SemanticNodeView::Check { condition, reason } => {
                self.lower_check(condition, reason, node.span())
            }
            SemanticNodeView::TuplePack { inputs, output } => {
                let items = inputs.iter().map(|value| self.bound(*value)).collect();
                self.values.bind(output, Bound::Tuple(items));
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

    fn bound(&self, value: SemanticValueId) -> Bound {
        self.values.get(value)
    }
    fn prepared(&mut self, bound: &Bound) -> PreparedArg {
        if let Bound::Scalar {
            direct: Some(expression),
            ..
        } = bound
        {
            return PreparedArg::Index(*expression);
        }
        let (symbol, dtype, index) = bound.scalar();
        if index {
            let expression = match self.builder.arena().symbol_sort(symbol) {
                SymbolSort::Nat => self.builder.arena().nat_symbol(symbol),
                SymbolSort::Int => {
                    let value = self.builder.arena().int_symbol(symbol);
                    self.builder.arena().nat_from_int(value)
                }
                SymbolSort::Scalar(_) => panic!("index binding has scalar symbol sort"),
            };
            PreparedArg::Index(expression)
        } else {
            PreparedArg::Scalar(symbol, dtype)
        }
    }

    fn kernel_arg(kernel: &mut PortableBuilder<'_, B>, arg: PreparedArg) -> PortableValue {
        match arg {
            PreparedArg::Index(value) => kernel.nat_arg(value),
            PreparedArg::Scalar(symbol, dtype) => kernel.scalar_arg(symbol, dtype),
        }
    }

    fn output_target(&mut self, value: SemanticValueId) -> Bound {
        match self.function.value(value).ty.clone() {
            SemanticType::Tensor(_) => Bound::Tensor(self.builder.portable_allocate_tensor(value)),
            SemanticType::Scalar(_) | SemanticType::Index { .. } => {
                let ScalarPublication::Scalar(slot) = self.builder.portable_publish(value) else {
                    panic!("scalar has range publication")
                };
                Bound::Scalar {
                    symbol: slot.symbol,
                    dtype: slot.dtype,
                    index: slot.sort == SymbolSort::Nat,
                    direct: None,
                    slot: Some(slot),
                }
            }
            SemanticType::Range { .. } => {
                let ScalarPublication::Range { start, end } = self.builder.portable_publish(value)
                else {
                    panic!("range has scalar publication")
                };
                Bound::Range {
                    start: Box::new(Bound::Scalar {
                        symbol: start.symbol,
                        dtype: start.dtype,
                        index: true,
                        direct: None,
                        slot: Some(start),
                    }),
                    end: Box::new(Bound::Scalar {
                        symbol: end.symbol,
                        dtype: end.dtype,
                        index: true,
                        direct: None,
                        slot: Some(end),
                    }),
                }
            }
            SemanticType::Tuple(_) => panic!("tuple survived semantic leaf normalization"),
            SemanticType::Opaque { .. } => panic!("opaque value escaped a checked portable body"),
            SemanticType::Void => Bound::Unit,
        }
    }

    fn assign(&mut self, source: &Bound, target: &Bound) {
        match (source, target) {
            (Bound::Tensor(source), Bound::Tensor(target)) if source != target => {
                self.copy_tensor(*source, *target);
            }
            (Bound::Tensor(_), Bound::Tensor(_)) | (Bound::Unit, Bound::Unit) => {}
            (
                Bound::Scalar { .. },
                Bound::Scalar {
                    slot: Some(slot), ..
                },
            ) => {
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
    fn copy_tensor(&mut self, source: AnyBufferView, destination: AnyBufferView) {
        let source_layout = self.builder.portable_layout(source);
        let destination_layout = self.builder.portable_layout(destination);
        if source_layout.contiguous && destination_layout.contiguous {
            self.builder.schedule().copy_any(source, destination);
            return;
        }
        assert_eq!(
            source.representation, destination.representation,
            "checked tensor copy changed representation"
        );
        assert!(
            matches!(
                registry::representation_info(source.representation).kind,
                RepresentationKind::Dense(_)
            ),
            "a transformed decode-only representation cannot be a copy destination"
        );
        let axes = destination_layout.extents;
        let parallel_domain = if self.mode != SemanticMode::AuthoredBackend {
            let extent = self.builder.arena().nat_product(&axes);
            Some(self.independent_domain(extent, "portable tensor copy workgroup size"))
        } else {
            None
        };
        let logical_zero = self.builder.arena().nat(0);
        let mut kernel = self.builder.portable_kernel();
        let source = kernel.arg_view(source, false);
        let destination = kernel.arg_view(destination, true);
        let mut logical_base = None;
        if let Some(domain) = parallel_domain {
            let (linear, base) = logical_global_id(&mut kernel, true, logical_zero);
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
                    let value = kernel.read(source, &index);
                    kernel.write(destination, &index, value);
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
                    let value = kernel.read(source, index);
                    kernel.write(destination, index, value);
                },
            );
        }
        let kernel = kernel.close();
        if let Some(domain) = parallel_domain {
            self.builder
                .schedule()
                .launch_semantic(kernel, domain, logical_base);
        } else {
            self.builder.schedule().launch_sequential(kernel);
        }
    }

    fn emit_scalar(
        &mut self,
        primitive: &PrimitiveId,
        inputs: &[SemanticValueId],
        out: SemanticValueId,
    ) {
        let index_output = matches!(self.function.value(out).ty, SemanticType::Index { .. });
        let args = inputs
            .iter()
            .map(|value| self.prepared(&self.bound(*value)))
            .collect::<Vec<_>>();
        let target = self.output_target(out);
        let Bound::Scalar {
            slot: Some(slot), ..
        } = target
        else {
            panic!("scalar primitive output is not scalar")
        };
        let symbolic = if let PrimitiveId::Symbolic(expression) = primitive {
            Some(capture_expr(
                self.builder.arena(),
                AnyExpr::Int(*expression),
                &self.values,
            ))
        } else {
            None
        };
        let mut kernel = self.builder.portable_kernel();
        let args = args
            .into_iter()
            .map(|arg| Self::kernel_arg(&mut kernel, arg))
            .collect::<Vec<_>>();
        let value = if let Some(expression) = symbolic {
            lower_captured_expr(&mut kernel, &expression)
        } else {
            lower_scalar_primitive(&mut kernel, primitive, &args, &self.function.value(out).ty)
        };
        let index_storage = ValueType::Scalar(DType::U32);
        let value = if index_output && value.ty != index_storage {
            kernel.cast(value, index_storage)
        } else {
            value
        };
        let destination = kernel.result_slot(slot);
        kernel.store_slot(destination, value);
        let kernel = kernel.close();
        self.builder.schedule().launch_sequential(kernel);
        self.values.bind(out, target);
    }

    fn lower_primitive(
        &mut self,
        primitive: &PrimitiveId,
        inputs: &[SemanticValueId],
        output: SemanticValueId,
    ) {
        if matches!(primitive, PrimitiveId::Cast(DType::U32)) {
            if let SemanticType::Index { bound } = self.function.value(output).ty {
                let one = self.builder.arena().nat(1);
                let direct = self.builder.arena().nat_sub(bound, one);
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
                    let (symbol, _, _) = self.bound(inputs[0]).scalar();
                    self.values.bind(
                        output,
                        Bound::Scalar {
                            symbol,
                            dtype: DType::U32,
                            index: true,
                            direct: Some(direct),
                            slot: None,
                        },
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
                self.values.bind(output, value);
            }
            PrimitiveId::RangeStart | PrimitiveId::RangeEnd => {
                let Bound::Range { start, end } = self.bound(inputs[0]) else {
                    panic!("range endpoint input is not a range")
                };
                self.values.bind(
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
            _ => self.emit_scalar(primitive, inputs, output),
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

        let scalar_target = match signature.result {
            registry::IntrinsicResultType::Scalar(_) => Some(self.output_target(output)),
            _ => None,
        };
        let owned_target = match signature.result {
            registry::IntrinsicResultType::Owned { .. } => {
                Some(self.builder.portable_allocate_tensor(output))
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
        let axes = match &self.function.value(output).ty {
            SemanticType::Tensor(tensor) => tensor.axes.clone(),
            _ => panic!("whole-tensor intrinsic result is not a checked tensor"),
        };
        let one = self.builder.arena().nat(1);
        let zero = self.builder.arena().nat(0);
        let parallel_extent = self.builder.arena().nat_product(&axes);
        let target = self.builder.portable_target_ref();
        let requirements = B::semantic_intrinsic_requirements(
            target,
            self.builder.arena(),
            signature,
            parallel_extent,
        );
        let workgroup = requirements.required_workgroup.unwrap_or([one, one, one]);
        let participants = self.builder.arena().nat_product(&workgroup);
        let adjusted = self.builder.arena().nat_add(parallel_extent, participants);
        let adjusted = self.builder.arena().nat_sub(adjusted, one);
        let groups = self.builder.arena().nat_div(adjusted, participants);
        let empty =
            self.builder
                .arena()
                .nat_cmp(seismic_lang::expr::CmpOp::Eq, parallel_extent, zero);
        let domain = SegmentLaunchDomain {
            mode: requirements
                .required_mode
                .unwrap_or(LaunchMode::Independent),
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
                    let scalar = SemanticScalar {
                        value,
                        dtype,
                        index: false,
                        uniformity: registry::IntrinsicUniformity::Workgroup,
                    };
                    if constant {
                        SemanticIntrinsicOperand::Constant(scalar)
                    } else {
                        SemanticIntrinsicOperand::Scalar(scalar)
                    }
                }
                IntrinsicPrepared::Place(view, representation, rank, writable) => {
                    assert_eq!(view.representation, representation);
                    let place = kernel.arg_view(view, writable);
                    let place = SemanticPlace {
                        tensor: kernel.tensor(place),
                        representation,
                        rank,
                        writable,
                    };
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
            SemanticPlace {
                tensor: kernel.tensor(place),
                representation: view.representation,
                rank: destination_rank.expect("owned intrinsic has no destination rank"),
                writable: true,
            }
        });
        let call = SemanticIntrinsicCall {
            signature,
            operands: &operands,
            destination,
        };
        let mut sink = SemanticIntrinsicSink::open(&mut kernel, &call);
        B::lower_semantic_intrinsic(target, &domain, call, &mut sink);
        let result = sink.finish();
        match (result, scalar_target.as_ref()) {
            (
                SemanticIntrinsicResult::Scalar(value),
                Some(Bound::Scalar {
                    slot: Some(slot), ..
                }),
            ) => {
                let destination = kernel.result_slot(*slot);
                kernel.store_slot(destination, value.value);
            }
            (SemanticIntrinsicResult::Owned(_), None) | (SemanticIntrinsicResult::Void, None) => {}
            (SemanticIntrinsicResult::Opaque(_), _) => {
                panic!("opaque intrinsic results are not admitted by the active registry")
            }
            _ => panic!("intrinsic result differs from its checked registry signature"),
        }
        let kernel = kernel.close();
        self.builder
            .schedule()
            .launch_semantic(kernel, domain, None);
        match signature.result {
            registry::IntrinsicResultType::Scalar(_) => {
                self.values.bind(
                    output,
                    scalar_target.expect("scalar intrinsic has a target"),
                );
            }
            registry::IntrinsicResultType::Owned { .. } => {
                self.values.bind(
                    output,
                    Bound::Tensor(owned_target.expect("owned intrinsic has a target")),
                );
            }
            registry::IntrinsicResultType::Void => {
                self.values.bind(output, Bound::Unit);
            }
            registry::IntrinsicResultType::Opaque { .. } => {
                panic!("opaque intrinsic results are not admitted by the active registry")
            }
        }
    }

    fn lower_call(&mut self, id: NodeId, inputs: &[SemanticValueId], outputs: &[SemanticValueId]) {
        let arguments = inputs
            .iter()
            .map(|value| match self.bound(*value) {
                Bound::Tensor(view) => ValueBinding::View {
                    view,
                    layout: self.builder.portable_layout(view),
                },
                Bound::Scalar { symbol, .. } => ValueBinding::Scalar(symbol),
                Bound::Range { start, end } => ValueBinding::Range {
                    start: start.scalar().0,
                    end: end.scalar().0,
                },
                Bound::Tuple(_) => {
                    panic!("tuple call argument survived semantic leaf normalization")
                }
                Bound::Unit => panic!("void call argument is not valid"),
            })
            .collect::<Vec<_>>();
        let call = self.builder.splice_call(id, &arguments);
        call.schedule(&mut self.builder.schedule());
        for output in outputs {
            let binding = match self.builder.portable_binding(*output) {
                ValueBinding::View { view, .. } => Bound::Tensor(view),
                ValueBinding::Scalar(symbol) => match self.function.value(*output).ty {
                    SemanticType::Index { .. } => Bound::Scalar {
                        symbol,
                        dtype: DType::U32,
                        index: true,
                        direct: None,
                        slot: publication_slot(self.builder.portable_publish(*output)),
                    },
                    SemanticType::Scalar(dtype) => Bound::Scalar {
                        symbol,
                        dtype,
                        index: false,
                        direct: None,
                        slot: publication_slot(self.builder.portable_publish(*output)),
                    },
                    _ => panic!("call scalar binding has non-scalar type"),
                },
                ValueBinding::Range { start, end } => Bound::Range {
                    start: Box::new(Bound::Scalar {
                        symbol: start,
                        dtype: DType::U32,
                        index: true,
                        direct: None,
                        slot: range_slots(self.builder.portable_publish(*output)).0,
                    }),
                    end: Box::new(Bound::Scalar {
                        symbol: end,
                        dtype: DType::U32,
                        index: true,
                        direct: None,
                        slot: range_slots(self.builder.portable_publish(*output)).1,
                    }),
                },
            };
            self.values.bind(*output, binding);
        }
    }

    fn lower_copy(&mut self, input: SemanticValueId, output: SemanticValueId) {
        let source = self.bound(input).tensor();
        let destination = self.builder.portable_allocate_tensor(output);
        self.copy_tensor(source, destination);
        self.values.bind(output, Bound::Tensor(destination));
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
        let destination_view = self.builder.portable_allocate_tensor(output);
        let recipe = registry::representation_conversion_info(conversion);
        assert_eq!(source.representation, recipe.source);
        assert_eq!(destination_view.representation, recipe.destination);
        let SemanticType::Tensor(output_tensor) = &self.function.value(output).ty else {
            panic!("checked representation conversion result is not a tensor")
        };
        let RepresentationKind::Packed(layout) =
            &registry::representation_info(recipe.destination).kind
        else {
            panic!(
                "registered representation conversion destination is not resident packed storage"
            )
        };
        let mut packet_axes = output_tensor.axes.clone();
        let last = packet_axes
            .last_mut()
            .expect("checked packed representation has no packing axis");
        let group = self.builder.arena().nat(u64::from(layout.group));
        *last = self.builder.arena().nat_ceil_div(*last, group);
        let packet_count = self.builder.arena().nat_product(&packet_axes);
        let logical_zero = self.builder.arena().nat(0);
        let mut kernel = self.builder.portable_kernel();
        let source_place = kernel.arg_view(source, false);
        let source = kernel.tensor(source_place);
        let destination_place = kernel.representation_destination(destination_view);
        let destination = kernel.tensor(destination_place);
        let packet_count_arg = kernel.nat_arg(packet_count);
        let compiler_owned_base = self.mode != SemanticMode::AuthoredBackend;
        let (packet, logical_base) =
            logical_global_id(&mut kernel, compiler_owned_base, logical_zero);
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
                mode: LaunchMode::Independent,
                grid: [packet_count, one, one],
                workgroup: [one, one, one],
                empty,
                parallel_extent: packet_count,
            }
        } else {
            self.independent_domain(packet_count, "representation conversion workgroup size")
        };
        self.builder
            .schedule()
            .launch_semantic(kernel, domain, logical_base);
        self.values.bind(output, Bound::Tensor(destination_view));
    }

    fn lower_view(
        &mut self,
        base_value: SemanticValueId,
        transform: &ViewTransform,
        out: SemanticValueId,
    ) {
        let base = self.bound(base_value).tensor();
        let layout = self.builder.portable_layout(base);
        let tensor = match &self.function.value(out).ty {
            SemanticType::Tensor(tensor) => tensor,
            _ => panic!("view output is not tensor"),
        };
        let view = match transform {
            ViewTransform::Identity => {
                let zero = self.builder.arena().nat(0);
                self.builder
                    .portable_define_view(out, base, zero, layout.extents, layout.strides)
            }
            ViewTransform::Transpose { permutation } => {
                let extents = permutation
                    .iter()
                    .map(|axis| layout.extents[*axis as usize])
                    .collect();
                let strides = permutation
                    .iter()
                    .map(|axis| layout.strides[*axis as usize])
                    .collect();
                let zero = self.builder.arena().nat(0);
                self.builder
                    .portable_define_view(out, base, zero, extents, strides)
            }
            ViewTransform::Reshape { axes } => {
                let strides = dense_strides(self.builder.arena(), tensor.representation, axes);
                let zero = self.builder.arena().nat(0);
                self.builder
                    .portable_define_view(out, base, zero, axes.clone(), strides)
            }
            ViewTransform::Slice { axes } => self.lower_slice_view(out, base, axes),
            ViewTransform::Plane { .. } => {
                panic!("physical plane view escaped into a checked portable body")
            }
        };
        self.values.bind(out, Bound::Tensor(view));
    }

    fn lower_slice_view(
        &mut self,
        out: SemanticValueId,
        base: AnyBufferView,
        axes: &[SliceAxis],
    ) -> AnyBufferView {
        let layout = self.builder.portable_layout(base);
        let mut offset_units = self.builder.arena().nat(0);
        let mut extents = Vec::new();
        let mut strides = Vec::new();
        for (axis, selection) in axes.iter().enumerate() {
            let stride = layout.strides[axis];
            match selection {
                SliceAxis::Full => {
                    extents.push(layout.extents[axis]);
                    strides.push(stride);
                }
                SliceAxis::Point(value) => {
                    let point = self.scalar_ref_nat(value);
                    let term = self.builder.arena().nat_mul(point, stride);
                    offset_units = self.builder.arena().nat_add(offset_units, term);
                }
                SliceAxis::Range { start, end } => {
                    let start = start
                        .as_ref()
                        .map(|value| self.scalar_ref_nat(value))
                        .unwrap_or_else(|| self.builder.arena().nat(0));
                    let end = end
                        .as_ref()
                        .map(|value| self.scalar_ref_nat(value))
                        .unwrap_or(layout.extents[axis]);
                    let term = self.builder.arena().nat_mul(start, stride);
                    offset_units = self.builder.arena().nat_add(offset_units, term);
                    extents.push(self.builder.arena().nat_sub(end, start));
                    strides.push(stride);
                }
            }
        }
        let unit = representation_unit_bytes(base.representation);
        let bytes = self.builder.arena().nat(unit);
        let offset = self.builder.arena().nat_mul(offset_units, bytes);
        self.builder
            .portable_define_view(out, base, offset, extents, strides)
    }

    fn scalar_ref_nat(&mut self, value: &ScalarRef) -> NatExpr {
        match value {
            ScalarRef::Static(value) => *value,
            ScalarRef::Value(value) => match self.prepared(&self.bound(*value)) {
                PreparedArg::Index(value) => value,
                PreparedArg::Scalar(_, _) => panic!("slice index is not an index"),
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
            .map(|value| self.prepared(&self.bound(*value)))
            .collect::<Vec<_>>();
        let target = self.output_target(output);
        let Bound::Scalar {
            slot: Some(slot), ..
        } = target
        else {
            panic!("element read output is not scalar")
        };
        let mut kernel = self.builder.portable_kernel();
        let place = kernel.arg_view(view, false);
        let indices = indices
            .into_iter()
            .map(|arg| {
                let value = Self::kernel_arg(&mut kernel, arg);
                portable_index(&mut kernel, value)
            })
            .collect::<Vec<_>>();
        let value = kernel.read(place, &indices);
        let destination = kernel.result_slot(slot);
        kernel.store_slot(destination, value);
        let kernel = kernel.close();
        self.builder.schedule().launch_sequential(kernel);
        self.values.bind(output, target);
    }
    fn lower_write_like(
        &mut self,
        place: SemanticValueId,
        indices: &[SemanticValueId],
        value: SemanticValueId,
        output: SemanticValueId,
        atomic: Option<seismic_lang::intrinsics::AtomicOp>,
    ) {
        let base = self.bound(place).tensor();
        let mut args = indices
            .iter()
            .map(|value| self.prepared(&self.bound(*value)))
            .collect::<Vec<_>>();
        args.push(self.prepared(&self.bound(value)));
        let mut kernel = self.builder.portable_kernel();
        let place = kernel.arg_view(base, true);
        let value = Self::kernel_arg(&mut kernel, args.pop().expect("write has no value"));
        let destination_type = value_type(element_dtype(base.representation));
        let value = if value.ty != destination_type {
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
            Some(op) => kernel.atomic(op, place, &indices, value),
            None => kernel.write(place, &indices, value),
        }
        let kernel = kernel.close();
        self.builder.schedule().launch_sequential(kernel);
        self.values.bind(output, Bound::Tensor(base));
    }

    fn lower_extent(&mut self, tensor: SemanticValueId, axis: u32, output: SemanticValueId) {
        let view = self.bound(tensor).tensor();
        let target = self.output_target(output);
        let Bound::Scalar {
            slot: Some(slot), ..
        } = target
        else {
            panic!("extent output is not index")
        };
        let mut kernel = self.builder.portable_kernel();
        let place = kernel.arg_view(view, false);
        let value = kernel.extent(place, axis);
        // Extents are index-domain values inside a kernel, while semantic
        // scalar results cross the schedule cut through their declared ABI
        // dtype (normally checked `i32`). Make that representation boundary
        // explicit before publishing the value.
        let storage = ValueType::Scalar(slot.dtype);
        let value = if value.ty != storage {
            kernel.cast(value, storage)
        } else {
            value
        };
        let destination = kernel.result_slot(slot);
        kernel.store_slot(destination, value);
        let kernel = kernel.close();
        self.builder.schedule().launch_sequential(kernel);
        self.values.bind(output, target);
    }

    fn lower_check(
        &mut self,
        condition_value: SemanticValueId,
        reason: &CheckReason,
        span: seismic_lang::span::Span,
    ) {
        let arg = self.prepared(&self.bound(condition_value));
        let slot = self.builder.schedule().temporary_bool();
        let mut kernel = self.builder.portable_kernel();
        let condition = Self::kernel_arg(&mut kernel, arg);
        let reason = match reason {
            CheckReason::IndexBound => "index out of bounds".into(),
            CheckReason::RangeOrder => "range endpoints out of order".into(),
            CheckReason::DivideByZero => "integer division by zero".into(),
            CheckReason::SignedDivisionOverflow => "signed integer division overflow".into(),
            CheckReason::Custom(text) => text.clone(),
        };
        let destination = kernel.result_slot(slot);
        kernel.store_slot(destination, condition);
        let kernel = kernel.close();
        self.builder.schedule().launch_sequential(kernel);
        self.builder.schedule().check_any(
            slot,
            CheckSite {
                reason,
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
        let destination = self.bound(destination_value).tensor();
        let source = self.bound(value).tensor();
        self.copy_tensor(source, destination);
        let base = match &self.function.value(destination_value).ty {
            SemanticType::Tensor(TensorSemantics {
                storage: TensorStorage::View { base, .. },
                ..
            }) => self.bound(*base).tensor(),
            _ => panic!("store destination is not an explicit writable view"),
        };
        self.values.bind(output, Bound::Tensor(base));
    }

    fn lower_elementwise(
        &mut self,
        primitive: &PrimitiveId,
        inputs: &[SemanticValueId],
        out: SemanticValueId,
    ) {
        let destination = self.builder.portable_allocate_tensor(out);
        let axes = match &self.function.value(out).ty {
            SemanticType::Tensor(tensor) => tensor.axes.clone(),
            _ => panic!("elementwise output is not tensor"),
        };
        let input_bounds = inputs
            .iter()
            .map(|value| self.bound(*value))
            .collect::<Vec<_>>();
        let scalar_args = input_bounds
            .iter()
            .map(|bound| match bound {
                Bound::Scalar { .. } => Some(self.prepared(bound)),
                _ => None,
            })
            .collect::<Vec<_>>();
        let input_axes = inputs
            .iter()
            .map(|value| match &self.function.value(*value).ty {
                SemanticType::Tensor(t) => Some(t.axes.clone()),
                _ => None,
            })
            .collect::<Vec<_>>();
        let broadcast = input_axes
            .iter()
            .map(|axes| {
                axes.as_ref().map(|axes| {
                    axes.iter()
                        .map(|axis| {
                            matches!(
                                self.builder.arena().view(AnyExpr::Nat(*axis)),
                                NodeView::NatConst(1)
                            )
                        })
                        .collect::<Vec<_>>()
                })
            })
            .collect::<Vec<_>>();
        let parallel_domain = if self.mode != SemanticMode::AuthoredBackend {
            let extent = self.builder.arena().nat_product(&axes);
            Some(self.independent_domain(extent, "portable elementwise workgroup size"))
        } else {
            None
        };
        let logical_zero = self.builder.arena().nat(0);
        let mut kernel = self.builder.portable_kernel();
        let output = kernel.arg_view(destination, true);
        let places = input_bounds
            .iter()
            .map(|bound| match bound {
                Bound::Tensor(view) => Some(kernel.arg_view(*view, false)),
                _ => None,
            })
            .collect::<Vec<_>>();
        let scalars = scalar_args
            .into_iter()
            .map(|arg| arg.map(|arg| Self::kernel_arg(&mut kernel, arg)))
            .collect::<Vec<_>>();
        let mut emit = |kernel: &mut PortableBuilder<'_, B>, index: &[PortableValue]| {
            let mut args = Vec::new();
            for (((place, scalar), source_axes), broadcast) in
                places.iter().zip(&scalars).zip(&input_axes).zip(&broadcast)
            {
                if let Some(place) = place {
                    let source_axes = source_axes.as_ref().expect("tensor input has no axes");
                    let skip = index.len() - source_axes.len();
                    let flags = broadcast
                        .as_ref()
                        .expect("tensor input has no broadcast map");
                    let source_index = index[skip..]
                        .iter()
                        .zip(flags)
                        .map(|(value, broadcast)| {
                            if *broadcast {
                                kernel.index_constant(0)
                            } else {
                                *value
                            }
                        })
                        .collect::<Vec<_>>();
                    args.push(kernel.read(*place, &source_index));
                } else {
                    args.push(scalar.expect("elementwise input is neither tensor nor scalar"));
                }
            }
            let value = lower_scalar_primitive(
                kernel,
                primitive,
                &args,
                &SemanticType::Scalar(element_dtype(destination.representation)),
            );
            kernel.write(output, index, value);
        };
        let mut logical_base = None;
        if let Some(domain) = parallel_domain {
            let (linear, base) = logical_global_id(&mut kernel, true, logical_zero);
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
                    emit(kernel, &index);
                    Vec::new()
                },
                |_| Vec::new(),
            );
        } else {
            nested(&mut kernel, &axes, 0, &mut Vec::new(), &mut emit);
        }
        let kernel = kernel.close();
        if let Some(domain) = parallel_domain {
            self.builder
                .schedule()
                .launch_semantic(kernel, domain, logical_base);
        } else {
            self.builder.schedule().launch_sequential(kernel);
        }
        self.values.bind(out, Bound::Tensor(destination));
    }

    fn stream_tensor_plan(&mut self, value: SemanticValueId) -> StreamTensorPlan {
        let SemanticType::Tensor(tensor_type) = &self.function.value(value).ty else {
            panic!("streamed tensor value has non-tensor semantics")
        };
        let axes = tensor_type.axes.clone();
        if !self.streamed_values.contains(&value) {
            return StreamTensorPlan {
                axes,
                value: StreamTensorPlanValue::Physical(self.bound(value).tensor()),
            };
        }

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
                let input_axes = inputs
                    .iter()
                    .map(|input| match input {
                        StreamBoundPlan::Tensor(tensor) => Some(tensor.axes.clone()),
                        StreamBoundPlan::Scalar(_) => None,
                    })
                    .collect();
                StreamTensorPlanValue::Elementwise {
                    primitive: primitive.clone(),
                    inputs,
                    input_axes,
                    output: SemanticType::Scalar(
                        registry::representation_info(tensor_type.representation).decoded,
                    ),
                }
            }
            SemanticNodeView::View {
                base,
                transform,
                output,
            } => {
                assert_eq!(output, value, "streamed view output mismatch");
                let base = Box::new(self.stream_tensor_plan(base));
                match transform {
                    ViewTransform::Identity => return *base,
                    ViewTransform::Transpose { permutation } => StreamTensorPlanValue::View {
                        base,
                        transform: StreamViewPlan::Transpose(permutation.clone()),
                    },
                    ViewTransform::Reshape { .. } => StreamTensorPlanValue::View {
                        base,
                        transform: StreamViewPlan::Reshape,
                    },
                    ViewTransform::Slice { axes } => {
                        let zero = self.builder.arena().nat(0);
                        let axes = axes
                            .iter()
                            .map(|axis| match axis {
                                SliceAxis::Full => StreamSliceAxisPlan::Full,
                                SliceAxis::Point(value) => {
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
                        StreamTensorPlanValue::View {
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
        StreamTensorPlan { axes, value: plan }
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
        input: SemanticValueId,
        out: SemanticValueId,
    ) {
        if self.streamed_values.contains(&input) {
            let source = self.stream_tensor_plan(input);
            self.lower_streamed_reduce(op, axis, input, out, source);
            return;
        }
        let source = self.bound(input).tensor();
        let destination = self.builder.portable_allocate_tensor(out);
        let input_axes = match &self.function.value(input).ty {
            SemanticType::Tensor(t) => t.axes.clone(),
            _ => panic!("reduction input is not tensor"),
        };
        let output_axes = match &self.function.value(out).ty {
            SemanticType::Tensor(t) => t.axes.clone(),
            _ => panic!("reduction output is not tensor"),
        };
        let input_dtype = element_dtype(source.representation);
        let schema = reduce_schema(op, input_dtype);
        let parallel_domain = if self.mode != SemanticMode::AuthoredBackend {
            let extent = self.builder.arena().nat_product(&output_axes);
            Some(self.independent_domain(extent, "portable reduction workgroup size"))
        } else {
            None
        };
        let logical_zero = self.builder.arena().nat(0);
        let mut kernel = self.builder.portable_kernel();
        let input = kernel.arg_view(source, false);
        let output = kernel.arg_view(destination, true);
        let mut emit = |kernel: &mut PortableBuilder<'_, B>, outer: &[PortableValue]| {
            let zero = kernel.index_constant(0);
            let end = kernel.nat_arg(input_axes[axis as usize]);
            let first_index = reduction_index(outer, axis as usize, zero);
            let first = kernel.read(input, &first_index);
            let first = if first.ty == value_type(schema.accumulator) {
                first
            } else {
                kernel.cast(first, value_type(schema.accumulator))
            };
            let initial = match op {
                ReduceOp::Sum => zero_of(kernel, first.ty.clone()),
                ReduceOp::Max | ReduceOp::Min | ReduceOp::Argmax => first,
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
            let result = kernel.repeat(start, end, carries, |kernel, binder, carry| {
                let index = reduction_index(outer, axis as usize, binder);
                let value = kernel.read(input, &index);
                let value = if value.ty == value_type(schema.accumulator) {
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
            kernel.write(output, outer, value);
        };
        let mut logical_base = None;
        if let Some(domain) = parallel_domain {
            let (linear, base) = logical_global_id(&mut kernel, true, logical_zero);
            logical_base = base;
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
        } else {
            nested(&mut kernel, &output_axes, 0, &mut Vec::new(), &mut emit);
        }
        let kernel = kernel.close();
        if let Some(domain) = parallel_domain {
            self.builder
                .schedule()
                .launch_semantic(kernel, domain, logical_base);
        } else {
            self.builder.schedule().launch_sequential(kernel);
        }
        self.values.bind(out, Bound::Tensor(destination));
    }

    fn lower_streamed_reduce(
        &mut self,
        op: ReduceOp,
        axis: u32,
        input_value: SemanticValueId,
        out: SemanticValueId,
        source_plan: StreamTensorPlan,
    ) {
        let destination = self.builder.portable_allocate_tensor(out);
        let input_axes = source_plan.axes.clone();
        let output_axes = match &self.function.value(out).ty {
            SemanticType::Tensor(t) => t.axes.clone(),
            _ => panic!("reduction output is not tensor"),
        };
        let input_dtype = match &self.function.value(input_value).ty {
            SemanticType::Tensor(t) => registry::representation_info(t.representation).decoded,
            _ => panic!("streamed reduction input is not tensor"),
        };
        let schema = reduce_schema(op, input_dtype);
        let extent = self.builder.arena().nat_product(&output_axes);
        let domain = self.independent_domain(extent, "portable reduction workgroup size");
        let logical_zero = self.builder.arena().nat(0);
        let mut kernel = self.builder.portable_kernel();
        let source = instantiate_stream_tensor(&mut kernel, &source_plan);
        let output = kernel.arg_view(destination, true);
        let emit = |kernel: &mut PortableBuilder<'_, B>, outer: &[PortableValue]| {
            let zero = kernel.index_constant(0);
            let end = kernel.nat_arg(input_axes[axis as usize]);
            let first_index = reduction_index(outer, axis as usize, zero);
            let first = source.read(kernel, &first_index);
            let first = if first.ty == value_type(schema.accumulator) {
                first
            } else {
                kernel.cast(first, value_type(schema.accumulator))
            };
            let initial = match op {
                ReduceOp::Sum => zero_of(kernel, first.ty.clone()),
                ReduceOp::Max | ReduceOp::Min | ReduceOp::Argmax => first,
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
            let result = kernel.repeat(start, end, carries, |kernel, binder, carry| {
                let index = reduction_index(outer, axis as usize, binder);
                let value = source.read(kernel, &index);
                let value = if value.ty == value_type(schema.accumulator) {
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
            kernel.write(output, outer, value);
        };
        let (linear, logical_base) = logical_global_id(&mut kernel, true, logical_zero);
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
        let kernel = kernel.close();
        self.builder
            .schedule()
            .launch_semantic(kernel, domain, logical_base);
        self.values.bind(out, Bound::Tensor(destination));
    }

    fn lower_if(
        &mut self,
        condition_value: SemanticValueId,
        capture_values: &[SemanticValueId],
        outputs: &[SemanticValueId],
        then_region: RegionId,
        else_region: RegionId,
    ) {
        let condition_bound = self.bound(condition_value);
        let condition = condition_expr(self.builder.arena(), &condition_bound);
        let targets = outputs
            .iter()
            .map(|value| self.output_target(*value))
            .collect::<Vec<_>>();
        let captures = capture_values
            .iter()
            .map(|value| self.bound(*value))
            .collect::<Vec<_>>();
        let function = self.function;
        let then_values = self.values.clone();
        let else_values = self.values.clone();
        let then_targets = targets.clone();
        let else_targets = targets.clone();
        let mode = self.mode;
        let streamed_values = self.streamed_values.clone();
        self.builder
            .portable_branch(
                condition,
                |builder| {
                    let mut child = Lowerer {
                        function,
                        builder,
                        values: then_values,
                        mode,
                        streamed_values: streamed_values.clone(),
                    };
                    child.bind_region_parameters(then_region, &captures);
                    child.lower_region(then_region);
                    child.assign_region_results(then_region, &then_targets);
                    Ok::<(), ()>(())
                },
                |builder| {
                    let mut child = Lowerer {
                        function,
                        builder,
                        values: else_values,
                        mode,
                        streamed_values: streamed_values.clone(),
                    };
                    child.bind_region_parameters(else_region, &captures);
                    child.lower_region(else_region);
                    child.assign_region_results(else_region, &else_targets);
                    Ok::<(), ()>(())
                },
            )
            .unwrap_or_else(|()| unreachable!());
        for (output, target) in outputs.iter().zip(targets) {
            self.values.bind(*output, target);
        }
    }

    fn lower_loop(
        &mut self,
        kind: LoopKind,
        start_value: SemanticValueId,
        end_value: SemanticValueId,
        capture_values: &[SemanticValueId],
        body: RegionId,
        carries: &[seismic_lang::entry::Carry],
        written_places: &[SemanticValueId],
    ) {
        if kind == LoopKind::Parallel {
            self.lower_parallel_segment(
                start_value,
                end_value,
                capture_values,
                body,
                carries,
                written_places,
            );
            return;
        }
        let start_bound = self.bound(start_value);
        let end_bound = self.bound(end_value);
        let start = index_expr(self.builder.arena(), &start_bound);
        let end = index_expr(self.builder.arena(), &end_bound);
        let captures = capture_values
            .iter()
            .map(|value| self.bound(*value))
            .collect::<Vec<_>>();
        let carry_targets = carries
            .iter()
            .map(|carry| (carry.clone(), self.output_target(carry.parameter)))
            .collect::<Vec<_>>();
        for (carry, target) in &carry_targets {
            let initial = self.bound(carry.initial);
            self.assign(&initial, target);
        }
        let function = self.function;
        let parent_values = self.values.clone();
        let mode = self.mode;
        let streamed_values = self.streamed_values.clone();
        self.builder
            .portable_repeat(start, end, |builder, binding| {
                let mut child = Lowerer {
                    function,
                    builder,
                    values: parent_values,
                    mode,
                    streamed_values: streamed_values.clone(),
                };
                child.bind_region_parameters(body, &captures);
                let RegionKind::LoopBody { binder_value, .. } = function.region(body).kind() else {
                    panic!("loop body has wrong region kind")
                };
                child.values.bind(
                    *binder_value,
                    Bound::Scalar {
                        symbol: binding.symbol,
                        dtype: DType::U32,
                        index: true,
                        direct: None,
                        slot: None,
                    },
                );
                for (carry, target) in &carry_targets {
                    child.values.rebind(carry.parameter, target.clone());
                }
                child.lower_region(body);
                for (carry, target) in &carry_targets {
                    let yielded = child.bound(carry.yielded);
                    child.assign(&yielded, target);
                }
                Ok::<(), ()>(())
            })
            .unwrap_or_else(|()| unreachable!());
        for (carry, target) in carry_targets {
            self.values.bind(carry.result, target);
        }
    }

    fn lower_parallel_segment(
        &mut self,
        start_value: SemanticValueId,
        end_value: SemanticValueId,
        capture_values: &[SemanticValueId],
        body: RegionId,
        carries: &[seismic_lang::entry::Carry],
        _written_places: &[SemanticValueId],
    ) {
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
        let mut required_mode = None;
        let mut required_workgroup = None;
        for intrinsic in &intrinsics {
            let signature = registry::intrinsic_signature(*intrinsic);
            let requirements =
                B::semantic_intrinsic_requirements(target, self.builder.arena(), signature, extent);
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
        let adjusted = self.builder.arena().nat_add(extent, participants);
        let adjusted = self.builder.arena().nat_sub(adjusted, one);
        let groups = self.builder.arena().nat_div(adjusted, participants);
        let empty = self
            .builder
            .arena()
            .nat_cmp(seismic_lang::expr::CmpOp::Eq, extent, zero);
        let domain = SegmentLaunchDomain {
            mode: required_mode.unwrap_or(LaunchMode::Independent),
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
        let logical_zero = self.builder.arena().nat(0);
        let compiler_owned_base = self.mode != SemanticMode::AuthoredBackend;
        let separate_preflight = domain.mode != LaunchMode::Independent && !checks.is_empty();
        if separate_preflight {
            let check_statuses = checks
                .iter()
                .map(|_| self.builder.portable_preflight_status())
                .collect::<Vec<_>>();
            let mut allowed = BTreeSet::new();
            let mut visiting = BTreeSet::from([function.id()]);
            Self::collect_preflight_region(function, body, &helpers, &mut allowed, &mut visiting);
            let mut kernel = self.builder.portable_kernel();
            let mut values = BTreeMap::new();
            for (parameter, capture) in capture_parameters.iter().zip(&outer_captures) {
                values.insert(*parameter, segment_capture(&mut kernel, capture));
            }
            let start_arg = kernel.nat_arg(start);
            let (global, logical_base) =
                logical_global_id(&mut kernel, compiler_owned_base, logical_zero);
            let binder = kernel.binary(BinaryOp::Add, start_arg, global);
            values.insert(*binder_value, SegmentBound::Scalar(binder));
            let mut check_tensors = BTreeMap::new();
            for ((id, _, _), status) in checks.iter().zip(&check_statuses) {
                let place = kernel.arg_view(status.view, true);
                check_tensors.insert(*id, kernel.tensor(place));
            }
            let end_arg = kernel.nat_arg(end);
            let active = kernel.cmp(CmpOp::Lt, binder, end_arg);
            kernel.branch(
                active,
                |kernel| {
                    let mut segment = SegmentLowerer {
                        function,
                        kernel,
                        values,
                        helpers: &helpers,
                        checks: &check_tensors,
                        target,
                        domain,
                        nested_safe: None,
                        allowed: Some(&allowed),
                        assume_checks: false,
                    };
                    let _ = segment.lower_region(body);
                    Vec::new()
                },
                |_| Vec::new(),
            );
            let kernel = kernel.close();
            self.builder
                .schedule()
                .launch_semantic(kernel, domain, logical_base);
            for ((_, reason, span), status) in checks.iter().zip(check_statuses) {
                self.builder.portable_finish_preflight(
                    status,
                    CheckSite {
                        reason: check_reason_text(reason),
                        path: self.function.name().to_owned(),
                        line: span.start,
                    },
                );
            }
        }

        let check_statuses = (!separate_preflight)
            .then(|| {
                checks
                    .iter()
                    .map(|_| self.builder.portable_preflight_status())
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();
        let mut kernel = self.builder.portable_kernel();
        let mut values = BTreeMap::new();
        for (parameter, capture) in capture_parameters.iter().zip(&outer_captures) {
            values.insert(*parameter, segment_capture(&mut kernel, capture));
        }
        let start_arg = kernel.nat_arg(start);
        let (global, logical_base) =
            logical_global_id(&mut kernel, compiler_owned_base, logical_zero);
        let binder = kernel.binary(BinaryOp::Add, start_arg, global);
        values.insert(*binder_value, SegmentBound::Scalar(binder));
        let mut check_tensors = BTreeMap::new();
        for ((id, _, _), status) in checks.iter().zip(&check_statuses) {
            let place = kernel.arg_view(status.view, true);
            check_tensors.insert(*id, kernel.tensor(place));
        }
        let end_arg = kernel.nat_arg(end);
        let active = kernel.cmp(CmpOp::Lt, binder, end_arg);
        kernel.branch(
            active,
            |kernel| {
                let mut segment = SegmentLowerer {
                    function,
                    kernel,
                    values,
                    helpers: &helpers,
                    checks: &check_tensors,
                    target,
                    domain,
                    nested_safe: None,
                    allowed: None,
                    assume_checks: separate_preflight,
                };
                let _ = segment.lower_region(body);
                Vec::new()
            },
            |_| Vec::new(),
        );
        let kernel = kernel.close();
        self.builder
            .schedule()
            .launch_semantic(kernel, domain, logical_base);
        for ((_, reason, span), status) in checks.into_iter().zip(check_statuses) {
            self.builder.portable_finish_preflight(
                status,
                CheckSite {
                    reason: check_reason_text(&reason),
                    path: self.function.name().to_owned(),
                    line: span.start,
                },
            );
        }
        assert!(
            carries.is_empty(),
            "checked parallel loops cannot carry reassigned state"
        );
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
                    event.access(),
                    seismic_lang::entry::AccessKind::Write(_)
                        | seismic_lang::entry::AccessKind::AtomicRmw { .. }
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
        checks: &mut Vec<(NodeId, CheckReason, seismic_lang::span::Span)>,
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
                    let helper = self.builder.authored_helper(family, B::NAME);
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
                SemanticNodeView::Check { reason, .. } => {
                    if !checks.iter().any(|(existing, _, _)| *existing == id) {
                        checks.push((id, reason.clone(), node.span()));
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

    fn collect_preflight_region(
        function: &'f SemanticFunction,
        region: RegionId,
        helpers: &BTreeMap<FamilyId, &'f SemanticFunction>,
        allowed: &mut BTreeSet<NodeId>,
        visiting: &mut BTreeSet<FunctionId>,
    ) -> bool {
        let mut contains_check = false;
        for (id, node) in function.nodes(region) {
            match node.view() {
                SemanticNodeView::Check { condition, .. } => {
                    allowed.insert(id);
                    Self::require_preflight_value(function, condition, helpers, allowed, visiting);
                    contains_check = true;
                }
                SemanticNodeView::If {
                    condition,
                    captures,
                    outputs: _,
                    then,
                    otherwise,
                } => {
                    let nested =
                        Self::collect_preflight_region(function, then, helpers, allowed, visiting)
                            | Self::collect_preflight_region(
                                function, otherwise, helpers, allowed, visiting,
                            );
                    if nested {
                        allowed.insert(id);
                        Self::require_preflight_value(
                            function, condition, helpers, allowed, visiting,
                        );
                        for capture in captures {
                            Self::require_preflight_value(
                                function, *capture, helpers, allowed, visiting,
                            );
                        }
                        contains_check = true;
                    }
                }
                SemanticNodeView::Loop {
                    start,
                    end,
                    captures,
                    body,
                    carries,
                    ..
                } => {
                    if Self::collect_preflight_region(function, body, helpers, allowed, visiting) {
                        allowed.insert(id);
                        for value in std::iter::once(&start)
                            .chain(std::iter::once(&end))
                            .chain(captures.iter())
                        {
                            Self::require_preflight_value(
                                function, *value, helpers, allowed, visiting,
                            );
                        }
                        for carry in carries {
                            Self::require_preflight_value(
                                function,
                                carry.initial,
                                helpers,
                                allowed,
                                visiting,
                            );
                            Self::require_preflight_value(
                                function,
                                carry.yielded,
                                helpers,
                                allowed,
                                visiting,
                            );
                        }
                        contains_check = true;
                    }
                }
                SemanticNodeView::Call { family, inputs, .. } => {
                    let helper = helpers[&family];
                    if visiting.insert(helper.id()) {
                        let nested = Self::collect_preflight_region(
                            helper,
                            helper.root(),
                            helpers,
                            allowed,
                            visiting,
                        );
                        visiting.remove(&helper.id());
                        if nested {
                            allowed.insert(id);
                            for input in inputs {
                                Self::require_preflight_value(
                                    function, *input, helpers, allowed, visiting,
                                );
                            }
                            contains_check = true;
                        }
                    }
                }
                _ => {}
            }
        }
        contains_check
    }

    fn require_preflight_value(
        function: &'f SemanticFunction,
        value: SemanticValueId,
        helpers: &BTreeMap<FamilyId, &'f SemanticFunction>,
        allowed: &mut BTreeSet<NodeId>,
        visiting: &mut BTreeSet<FunctionId>,
    ) {
        let ValueOrigin::Node(node) = function.value(value).origin else {
            return;
        };
        if !allowed.insert(node) {
            return;
        }
        let producer = function.node(node);
        assert!(
            producer.events().iter().all(|event| !matches!(
                event.access(),
                seismic_lang::entry::AccessKind::Write(_)
                    | seismic_lang::entry::AccessKind::AtomicRmw { .. }
            )),
            "checked safety predicate depends on an effectful write"
        );
        match producer.view() {
            SemanticNodeView::Intrinsic { .. }
            | SemanticNodeView::ElementWrite { .. }
            | SemanticNodeView::Store { .. }
            | SemanticNodeView::Atomic { .. }
            | SemanticNodeView::RepresentationConvert { .. } => {
                panic!("checked safety predicate depends on an unsafe/effectful operation")
            }
            SemanticNodeView::Call {
                family,
                inputs,
                outputs,
            } => {
                let helper = helpers[&family];
                for input in inputs {
                    Self::require_preflight_value(function, *input, helpers, allowed, visiting);
                }
                if visiting.insert(helper.id()) {
                    for (output, result) in outputs.iter().zip(helper.results()) {
                        if *output == value {
                            Self::require_preflight_value(
                                helper, *result, helpers, allowed, visiting,
                            );
                        }
                    }
                    visiting.remove(&helper.id());
                }
            }
            SemanticNodeView::If {
                condition,
                captures,
                outputs,
                then,
                otherwise,
            } => {
                Self::require_preflight_value(function, condition, helpers, allowed, visiting);
                for capture in captures {
                    Self::require_preflight_value(function, *capture, helpers, allowed, visiting);
                }
                for (index, output) in outputs.iter().enumerate() {
                    if *output == value {
                        Self::require_preflight_value(
                            function,
                            function.region(then).results()[index],
                            helpers,
                            allowed,
                            visiting,
                        );
                        Self::require_preflight_value(
                            function,
                            function.region(otherwise).results()[index],
                            helpers,
                            allowed,
                            visiting,
                        );
                    }
                }
            }
            SemanticNodeView::Loop {
                start,
                end,
                captures,
                carries,
                ..
            } => {
                for dependency in std::iter::once(&start)
                    .chain(std::iter::once(&end))
                    .chain(captures.iter())
                {
                    Self::require_preflight_value(
                        function,
                        *dependency,
                        helpers,
                        allowed,
                        visiting,
                    );
                }
                for carry in carries {
                    if carry.result == value {
                        Self::require_preflight_value(
                            function,
                            carry.initial,
                            helpers,
                            allowed,
                            visiting,
                        );
                        Self::require_preflight_value(
                            function,
                            carry.yielded,
                            helpers,
                            allowed,
                            visiting,
                        );
                    }
                }
            }
            _ => {
                for dependency in producer.dependencies() {
                    Self::require_preflight_value(function, dependency, helpers, allowed, visiting);
                }
            }
        }
    }

    fn segment_capture_plan(&mut self, bound: &Bound, writable: bool) -> SegmentCapture {
        match bound {
            Bound::Tensor(view) => SegmentCapture::Tensor {
                view: *view,
                writable,
            },
            Bound::Scalar { .. } => SegmentCapture::Scalar(self.prepared(bound)),
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
            self.values.bind(*parameter, capture.clone());
        }
    }
    fn assign_region_results(&mut self, region: RegionId, targets: &[Bound]) {
        let results = self.function.region(region).results();
        assert_eq!(results.len(), targets.len());
        for (result, target) in results.iter().zip(targets) {
            let source = self.bound(*result);
            self.assign(&source, target);
        }
    }

    fn publish_results(&mut self) {
        for result in self.function.results() {
            let source = self.bound(*result);
            let target = self.output_target(*result);
            self.assign(&source, &target);
            self.values.rebind(*result, target);
        }
    }
}

struct SegmentLowerer<'s, 'k, 'f, 'r, B: Backend> {
    function: &'f SemanticFunction,
    kernel: &'s mut PortableBuilder<'k, B>,
    values: BTreeMap<SemanticValueId, SegmentBound>,
    helpers: &'r BTreeMap<FamilyId, &'f SemanticFunction>,
    checks: &'r BTreeMap<NodeId, PortableTensor>,
    target: &'f crate::target::DeviceContract<B>,
    domain: SegmentLaunchDomain,
    nested_safe: Option<PortableValue>,
    allowed: Option<&'r BTreeSet<NodeId>>,
    assume_checks: bool,
}

#[derive(Clone)]
enum SegmentFrame<'f> {
    Nodes {
        function: &'f SemanticFunction,
        nodes: Vec<NodeId>,
    },
    Bind {
        outputs: Vec<SemanticValueId>,
        results: Vec<SemanticValueId>,
    },
}

impl<'s, 'k, 'f, 'r, B: Backend> SegmentLowerer<'s, 'k, 'f, 'r, B> {
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

    fn physical_tensor(&self, value: SemanticValueId) -> PortableTensor {
        match self.tensor(value).value {
            SegmentTensorValue::Physical(value) => value,
            _ => panic!("checked writable/capability tensor is not physical storage"),
        }
    }

    fn lower_region(&mut self, region: RegionId) -> PortableValue {
        let nodes = self
            .function
            .nodes(region)
            .map(|(id, _)| id)
            .collect::<Vec<_>>();
        self.lower_frames(vec![SegmentFrame::Nodes {
            function: self.function,
            nodes,
        }])
    }

    fn lower_frames(&mut self, mut frames: Vec<SegmentFrame<'f>>) -> PortableValue {
        let Some(frame) = frames.first().cloned() else {
            return self
                .kernel
                .constant(ConstantValue::Bool(true), ValueType::Bool);
        };
        frames.remove(0);
        let (function, id, rest) = match frame {
            SegmentFrame::Bind { outputs, results } => {
                for (output, result) in outputs.into_iter().zip(results) {
                    if let Some(bound) = self.values.get(&result).cloned() {
                        self.values.insert(output, bound);
                    } else {
                        assert!(
                            self.allowed.is_some(),
                            "checked segment result is not dominated"
                        );
                    }
                }
                return self.lower_frames(frames);
            }
            SegmentFrame::Nodes { function, nodes } => {
                let Some((&id, rest)) = nodes.split_first() else {
                    return self.lower_frames(frames);
                };
                (function, id, rest.to_vec())
            }
        };
        self.function = function;
        if !rest.is_empty() {
            frames.insert(
                0,
                SegmentFrame::Nodes {
                    function,
                    nodes: rest,
                },
            );
        }
        let node = function.node(id);
        if self.assume_checks && matches!(node.view(), SemanticNodeView::Check { .. }) {
            return self.lower_frames(frames);
        }
        if self.allowed.is_some_and(|allowed| !allowed.contains(&id)) {
            return self.lower_frames(frames);
        }
        if let SemanticNodeView::Check { condition, .. } = node.view() {
            let condition = self.scalar(condition);
            let status = self.checks[&id].clone();
            self.kernel.preflight_fail(&status, condition);
            let values = self.values.clone();
            let result = self.kernel.branch(
                condition,
                |kernel| {
                    let mut child = SegmentLowerer {
                        function,
                        kernel,
                        values,
                        helpers: self.helpers,
                        checks: self.checks,
                        target: self.target,
                        domain: self.domain,
                        nested_safe: None,
                        allowed: self.allowed,
                        assume_checks: self.assume_checks,
                    };
                    vec![child.lower_frames(frames)]
                },
                |kernel| vec![kernel.constant(ConstantValue::Bool(false), ValueType::Bool)],
            );
            return result[0];
        }

        if let SemanticNodeView::Call {
            family,
            inputs,
            outputs,
        } = node.view()
        {
            let helper = self.helpers[&family];
            assert_eq!(helper.parameters().len(), inputs.len());
            for (parameter, input) in helper.parameters().iter().zip(inputs) {
                self.values.insert(parameter.value, self.bound(*input));
            }
            frames.insert(
                0,
                SegmentFrame::Bind {
                    outputs: outputs.to_vec(),
                    results: helper.results().to_vec(),
                },
            );
            frames.insert(
                0,
                SegmentFrame::Nodes {
                    function: helper,
                    nodes: helper.nodes(helper.root()).map(|(id, _)| id).collect(),
                },
            );
            return self.lower_frames(frames);
        }

        if let SemanticNodeView::If {
            condition,
            captures,
            outputs,
            then,
            otherwise,
        } = node.view()
        {
            let condition = self.scalar(condition);
            let captures = captures
                .iter()
                .map(|value| self.bound(*value))
                .collect::<Vec<_>>();
            let values = self.values.clone();
            let branch_frames = |region: RegionId, mut continuation: Vec<SegmentFrame<'f>>| {
                continuation.insert(
                    0,
                    SegmentFrame::Bind {
                        outputs: outputs.to_vec(),
                        results: function.region(region).results().to_vec(),
                    },
                );
                continuation.insert(
                    0,
                    SegmentFrame::Nodes {
                        function,
                        nodes: function.nodes(region).map(|(id, _)| id).collect(),
                    },
                );
                continuation
            };
            let then_frames = branch_frames(then, frames.clone());
            let else_frames = branch_frames(otherwise, frames);
            let then_values = values.clone();
            let else_values = values;
            let result = self.kernel.branch(
                condition,
                |kernel| {
                    let mut child = SegmentLowerer {
                        function,
                        kernel,
                        values: then_values,
                        helpers: self.helpers,
                        checks: self.checks,
                        target: self.target,
                        domain: self.domain,
                        nested_safe: None,
                        allowed: self.allowed,
                        assume_checks: self.assume_checks,
                    };
                    bind_segment_region_parameters(&mut child.values, function, then, &captures);
                    vec![child.lower_frames(then_frames)]
                },
                |kernel| {
                    let mut child = SegmentLowerer {
                        function,
                        kernel,
                        values: else_values,
                        helpers: self.helpers,
                        checks: self.checks,
                        target: self.target,
                        domain: self.domain,
                        nested_safe: None,
                        allowed: self.allowed,
                        assume_checks: self.assume_checks,
                    };
                    bind_segment_region_parameters(
                        &mut child.values,
                        function,
                        otherwise,
                        &captures,
                    );
                    vec![child.lower_frames(else_frames)]
                },
            );
            return result[0];
        }

        self.nested_safe = None;
        self.lower_node(id);
        if let Some(safe) = self.nested_safe.take() {
            let function = self.function;
            let values = self.values.clone();
            let result = self.kernel.branch(
                safe,
                |kernel| {
                    let mut child = SegmentLowerer {
                        function,
                        kernel,
                        values,
                        helpers: self.helpers,
                        checks: self.checks,
                        target: self.target,
                        domain: self.domain,
                        nested_safe: None,
                        allowed: self.allowed,
                        assume_checks: self.assume_checks,
                    };
                    vec![child.lower_frames(frames)]
                },
                |kernel| vec![kernel.constant(ConstantValue::Bool(false), ValueType::Bool)],
            );
            result[0]
        } else {
            self.lower_frames(frames)
        }
    }

    fn lower_node(&mut self, id: NodeId) {
        let node = self.function.node(id);
        match node.view() {
            SemanticNodeView::Primitive {
                primitive,
                inputs,
                output,
            } => self.lower_primitive(primitive, inputs, output),
            SemanticNodeView::Intrinsic {
                intrinsic,
                inputs,
                output,
            } => self.lower_intrinsic(intrinsic, inputs, output),
            SemanticNodeView::Elementwise {
                primitive,
                inputs,
                output,
            } => self.lower_elementwise(primitive, inputs, output),
            SemanticNodeView::Reduce {
                op,
                axis,
                input,
                output,
                ..
            } => self.lower_reduce(op, axis, input, output),
            SemanticNodeView::View {
                base,
                transform,
                output,
            } => {
                let mut tensor = self.tensor(base);
                tensor = match transform {
                    ViewTransform::Identity => tensor,
                    ViewTransform::Plane { plane } => {
                        let SegmentTensorValue::Physical(physical) = tensor.value else {
                            panic!("checked plane view requires packed physical storage")
                        };
                        let physical = self.kernel.tensor_plane(physical, *plane);
                        SegmentTensor::physical(self.kernel, physical)
                    }
                    ViewTransform::Transpose { permutation } => SegmentTensor {
                        axes: permutation
                            .iter()
                            .map(|axis| tensor.axes[*axis as usize])
                            .collect(),
                        value: SegmentTensorValue::View {
                            base: Box::new(tensor),
                            transform: SegmentViewTransform::Transpose(permutation.clone()),
                        },
                    },
                    ViewTransform::Reshape { axes } => {
                        let axes = axes.iter().map(|axis| self.kernel.nat_arg(*axis)).collect();
                        SegmentTensor {
                            axes,
                            value: SegmentTensorValue::View {
                                base: Box::new(tensor),
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
                                SliceAxis::Point(value) => {
                                    SegmentSliceAxis::Point(self.scalar_ref(value))
                                }
                                SliceAxis::Range { start, end } => {
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
                            value: SegmentTensorValue::View {
                                base: Box::new(tensor),
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
                if value.ty != destination_type {
                    value = self.kernel.cast(value, destination_type);
                }
                self.kernel.tensor_write(&tensor, &indices, value);
                self.values.insert(
                    output,
                    SegmentBound::Tensor(SegmentTensor::physical(self.kernel, tensor)),
                );
            }
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
                    SegmentBound::Tensor(SegmentTensor::physical(self.kernel, tensor)),
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
            SemanticNodeView::Loop {
                start,
                end,
                captures,
                body,
                carries,
                ..
            } => self.lower_repeat(start, end, captures, body, carries),
            SemanticNodeView::Alloc { output } => self.lower_local_alloc(output),
            SemanticNodeView::Fill { value, output } => self.lower_local_fill(value, output),
            SemanticNodeView::Copy { input, output } => self.lower_local_copy(input, output),
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
                    && value.ty != ValueType::Index
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
                let value = lower_scalar_primitive(
                    self.kernel,
                    primitive,
                    &arguments,
                    &self.function.value(output).ty,
                );
                self.values.insert(output, SegmentBound::Scalar(value));
            }
        }
    }

    fn lower_elementwise(
        &mut self,
        primitive: &PrimitiveId,
        inputs: &[SemanticValueId],
        output: SemanticValueId,
    ) {
        let SemanticType::Tensor(output_type) = &self.function.value(output).ty else {
            panic!("checked elementwise output is not a tensor")
        };
        let axes = output_type
            .axes
            .iter()
            .map(|axis| self.kernel.nat_arg(*axis))
            .collect();
        let inputs = inputs
            .iter()
            .map(|value| self.bound(*value))
            .collect::<Vec<_>>();
        let input_axes = inputs
            .iter()
            .map(|input| match input {
                SegmentBound::Tensor(tensor) => Some(tensor.axes.clone()),
                _ => None,
            })
            .collect();
        self.values.insert(
            output,
            SegmentBound::Tensor(SegmentTensor {
                axes,
                value: SegmentTensorValue::Elementwise {
                    primitive: primitive.clone(),
                    inputs,
                    input_axes,
                    output: SemanticType::Scalar(
                        registry::representation_info(output_type.representation).decoded,
                    ),
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
    ) {
        let input_id = input;
        let input = self.tensor(input_id);
        let SemanticType::Tensor(input_type) = &self.function.value(input_id).ty else {
            unreachable!()
        };
        let mut axes = input.axes.clone();
        axes.remove(axis as usize);
        self.values.insert(
            output,
            SegmentBound::Tensor(SegmentTensor {
                axes,
                value: SegmentTensorValue::Reduce {
                    op,
                    axis: axis as usize,
                    input_dtype: registry::representation_info(input_type.representation).decoded,
                    input: Box::new(input),
                },
            }),
        );
    }

    fn lower_store(
        &mut self,
        destination: SemanticValueId,
        value: SemanticValueId,
        output: SemanticValueId,
    ) {
        let destination_tensor = self.physical_tensor(destination);
        let source = self.tensor(value);
        let axes = self.kernel.tensor_extents(&destination_tensor).to_vec();
        segment_copy_tensor(
            self.kernel,
            &source,
            &destination_tensor,
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

    fn local_tensor(&mut self, output: SemanticValueId) -> PortableTensor {
        let SemanticType::Tensor(tensor) = &self.function.value(output).ty else {
            panic!("checked local allocation output is not a tensor")
        };
        self.kernel.local_tensor(
            LaunchLocalKind::Participant,
            tensor.representation,
            tensor.axes.clone(),
        )
    }

    fn lower_local_alloc(&mut self, output: SemanticValueId) {
        let tensor = self.local_tensor(output);
        self.values.insert(
            output,
            SegmentBound::Tensor(SegmentTensor::physical(self.kernel, tensor)),
        );
    }

    fn lower_local_fill(
        &mut self,
        fill: seismic_lang::intrinsics::FillConstant,
        output: SemanticValueId,
    ) {
        let tensor = self.local_tensor(output);
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
            SegmentBound::Tensor(SegmentTensor::physical(self.kernel, tensor)),
        );
    }

    fn lower_local_copy(&mut self, input: SemanticValueId, output: SemanticValueId) {
        let source = self.tensor(input);
        let tensor = self.local_tensor(output);
        let axes = self.kernel.tensor_extents(&tensor).to_vec();
        segment_copy_tensor(self.kernel, &source, &tensor, &axes, 0, &mut Vec::new());
        self.values.insert(
            output,
            SegmentBound::Tensor(SegmentTensor::physical(self.kernel, tensor)),
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
                    SemanticIntrinsicOperand::Scalar(SemanticScalar {
                        value,
                        dtype,
                        index: value.ty == ValueType::Index,
                        uniformity: self.kernel.uniformity(value),
                    })
                }
                registry::OperandCategory::Constant(dtype) => {
                    let value = self.scalar(*input);
                    SemanticIntrinsicOperand::Constant(SemanticScalar {
                        value,
                        dtype,
                        index: value.ty == ValueType::Index,
                        uniformity: self.kernel.uniformity(value),
                    })
                }
                registry::OperandCategory::Readable {
                    representation,
                    rank,
                } => {
                    let tensor = self.physical_tensor(*input);
                    SemanticIntrinsicOperand::Readable(SemanticPlace {
                        tensor,
                        representation,
                        rank,
                        writable: false,
                    })
                }
                registry::OperandCategory::Writable {
                    representation,
                    rank,
                } => {
                    let tensor = self.physical_tensor(*input);
                    SemanticIntrinsicOperand::Writable(SemanticPlace {
                        tensor,
                        representation,
                        rank,
                        writable: true,
                    })
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
        B::lower_semantic_intrinsic(self.target, &self.domain, call, &mut sink);
        let result = sink.finish();
        let value = match result {
            SemanticIntrinsicResult::Scalar(value) => SegmentBound::Scalar(value.value),
            SemanticIntrinsicResult::Opaque(value) => SegmentBound::Opaque(value),
            SemanticIntrinsicResult::Void => SegmentBound::Unit,
            SemanticIntrinsicResult::Owned(_) => {
                panic!("enclosing-parallel intrinsic cannot return an owned tensor")
            }
        };
        self.values.insert(output, value);
    }

    fn lower_repeat(
        &mut self,
        start: SemanticValueId,
        end: SemanticValueId,
        captures: &[SemanticValueId],
        body: RegionId,
        carries: &[seismic_lang::entry::Carry],
    ) {
        let start = portable_index(self.kernel, self.scalar(start));
        let end = portable_index(self.kernel, self.scalar(end));
        let mut initial = carries
            .iter()
            .map(|carry| self.scalar(carry.initial))
            .collect::<Vec<_>>();
        initial.push(
            self.kernel
                .constant(ConstantValue::Bool(true), ValueType::Bool),
        );
        let parent = self.values.clone();
        let function = self.function;
        let parameters = function.region(body).parameters().to_vec();
        let capture_values = captures
            .iter()
            .map(|value| self.bound(*value))
            .collect::<Vec<_>>();
        let mut final_values = None;
        let results = self
            .kernel
            .repeat(start, end, initial, |kernel, binder, carried| {
                let otherwise_values = carried.clone();
                let (&incoming_safe, carried_values) = carried
                    .split_last()
                    .expect("repeat safety carry is present");
                let carried_values = carried_values.to_vec();
                kernel.branch(
                    incoming_safe,
                    |kernel| {
                        let mut child = SegmentLowerer {
                            function,
                            kernel,
                            values: parent,
                            helpers: self.helpers,
                            checks: self.checks,
                            target: self.target,
                            domain: self.domain,
                            nested_safe: None,
                            allowed: self.allowed,
                            assume_checks: self.assume_checks,
                        };
                        child
                            .values
                            .insert(parameters[0], SegmentBound::Scalar(binder));
                        for (parameter, capture) in parameters[1..].iter().zip(&capture_values) {
                            child.values.insert(*parameter, capture.clone());
                        }
                        for (carry, value) in carries.iter().zip(&carried_values) {
                            child
                                .values
                                .insert(carry.parameter, SegmentBound::Scalar(*value));
                        }
                        let safe = child.lower_region(body);
                        let mut next = carries
                            .iter()
                            .map(|carry| child.scalar(carry.yielded))
                            .collect::<Vec<_>>();
                        next.push(safe);
                        final_values = Some(child.values);
                        next
                    },
                    |_| otherwise_values,
                )
            });
        self.values = final_values.expect("checked repeat body was not lowered");
        let (&safe, results) = results
            .split_last()
            .expect("repeat safety result is present");
        for (carry, value) in carries.iter().zip(results) {
            self.values
                .insert(carry.result, SegmentBound::Scalar(*value));
        }
        self.nested_safe = Some(safe);
    }
}

#[derive(Clone, Copy, Debug)]
enum PreparedArg {
    Index(NatExpr),
    Scalar(seismic_lang::expr::SymbolId, DType),
}

#[derive(Clone, Copy)]
enum IntrinsicPrepared {
    Scalar(PreparedArg, DType, bool),
    Place(
        AnyBufferView,
        seismic_lang::ids::RepresentationId,
        u32,
        bool,
    ),
}

fn publication_slot(publication: ScalarPublication) -> Option<AnyScalarSlot> {
    match publication {
        ScalarPublication::Scalar(slot) => Some(slot),
        _ => panic!("expected scalar publication"),
    }
}

fn check_reason_text(reason: &CheckReason) -> String {
    match reason {
        CheckReason::IndexBound => "index out of bounds".into(),
        CheckReason::RangeOrder => "range endpoints out of order".into(),
        CheckReason::DivideByZero => "integer division by zero".into(),
        CheckReason::SignedDivisionOverflow => "signed integer division overflow".into(),
        CheckReason::Custom(text) => text.clone(),
    }
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

fn segment_capture<B: Backend>(
    kernel: &mut PortableBuilder<'_, B>,
    bound: &SegmentCapture,
) -> SegmentBound {
    match bound {
        SegmentCapture::Tensor { view, writable } => {
            let place = kernel.arg_view(*view, *writable);
            let tensor = kernel.tensor(place);
            SegmentBound::Tensor(SegmentTensor::physical(kernel, tensor))
        }
        SegmentCapture::Scalar(argument) => SegmentBound::Scalar(match argument {
            PreparedArg::Index(expression) => kernel.nat_arg(*expression),
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

fn segment_copy_tensor<B: Backend>(
    kernel: &mut PortableBuilder<'_, B>,
    source: &SegmentTensor,
    destination: &PortableTensor,
    axes: &[PortableValue],
    axis: usize,
    index: &mut Vec<PortableValue>,
) {
    if axis == axes.len() {
        let value = source.read(kernel, index);
        kernel.tensor_write(destination, index, value);
        return;
    }
    let zero = kernel.index_constant(0);
    let end = axes[axis];
    kernel.repeat(zero, end, Vec::new(), |kernel, binder, _| {
        index.push(binder);
        segment_copy_tensor(kernel, source, destination, axes, axis + 1, index);
        index.pop();
        Vec::new()
    });
}

fn segment_fill_tensor<B: Backend>(
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
    kernel.repeat(zero, end, Vec::new(), |kernel, binder, _| {
        index.push(binder);
        segment_fill_tensor(kernel, destination, axes, axis + 1, index, value);
        index.pop();
        Vec::new()
    });
}

fn unravel_index<B: Backend>(
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

fn range_slots(publication: ScalarPublication) -> (Option<AnyScalarSlot>, Option<AnyScalarSlot>) {
    match publication {
        ScalarPublication::Range { start, end } => (Some(start), Some(end)),
        _ => panic!("expected range publication"),
    }
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

fn condition_expr(arena: &mut ExprArena, bound: &Bound) -> seismic_lang::expr::BoolExpr {
    let (symbol, dtype, index) = bound.scalar();
    assert_eq!(dtype, DType::Bool);
    assert!(!index);
    let value = arena.scalar_symbol::<seismic_lang::expr::BoolScalar>(symbol);
    let yes = arena.scalar_const::<seismic_lang::expr::BoolScalar>(true);
    arena.scalar_cmp(seismic_lang::expr::CmpOp::Eq, value, yes)
}
fn index_expr(arena: &mut ExprArena, bound: &Bound) -> NatExpr {
    if let Bound::Scalar {
        direct: Some(expression),
        ..
    } = bound
    {
        return *expression;
    }
    let (symbol, _, index) = bound.scalar();
    assert!(index);
    match arena.symbol_sort(symbol) {
        SymbolSort::Nat => arena.nat_symbol(symbol),
        SymbolSort::Int => {
            let value = arena.int_symbol(symbol);
            arena.nat_from_int(value)
        }
        SymbolSort::Scalar(_) => panic!("index has scalar symbol"),
    }
}

fn nested<B: Backend>(
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
    kernel.repeat(start, end, Vec::new(), |kernel, binder, _| {
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
fn zero_of<B: Backend>(kernel: &mut PortableBuilder<'_, B>, ty: ValueType) -> PortableValue {
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

fn one_of<B: Backend>(kernel: &mut PortableBuilder<'_, B>, ty: ValueType) -> PortableValue {
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

fn portable_index<B: Backend>(
    kernel: &mut PortableBuilder<'_, B>,
    value: PortableValue,
) -> PortableValue {
    match value.ty {
        ValueType::Index => value,
        ValueType::Scalar(DType::I32 | DType::U32) => kernel.cast(value, ValueType::Index),
        _ => panic!("checked tensor index has a non-integer portable type"),
    }
}

fn scalar_bits<B: Backend>(
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

fn lower_scalar_primitive<B: Backend>(
    kernel: &mut PortableBuilder<'_, B>,
    primitive: &PrimitiveId,
    args: &[PortableValue],
    output: &SemanticType,
) -> PortableValue {
    let mut binary_args = || {
        let mut lhs = args[0];
        let mut rhs = args[1];
        match (&lhs.ty, &rhs.ty) {
            (ValueType::Index, ValueType::Scalar(DType::I32)) => {
                lhs = kernel.cast(lhs, ValueType::Scalar(DType::I32));
            }
            (ValueType::Scalar(DType::I32), ValueType::Index) => {
                rhs = kernel.cast(rhs, ValueType::Scalar(DType::I32));
            }
            _ => {}
        }
        (lhs, rhs)
    };
    match primitive {
        PrimitiveId::Constant(value) => lower_constant(kernel, *value, output),
        PrimitiveId::Unary(op) => match op {
            ast::UnaryOp::Neg => kernel.unary(UnaryOp::Neg, args[0]),
            ast::UnaryOp::Not => kernel.not(args[0]),
            ast::UnaryOp::BitNot => {
                let ones = match args[0].ty {
                    ValueType::Scalar(DType::I32) => {
                        kernel.constant(ConstantValue::I32(-1), args[0].ty.clone())
                    }
                    ValueType::Scalar(DType::U32) => {
                        kernel.constant(ConstantValue::U32(u32::MAX), args[0].ty.clone())
                    }
                    _ => panic!("checked bit-not has invalid type"),
                };
                kernel.bit(BitOp::Xor, args[0], ones)
            }
        },
        PrimitiveId::Binary(op) => {
            // Checked source arithmetic treats `index[N]` as a proved
            // non-negative integer when it meets an ordinary `i32` operand.
            // Portable IR keeps schedule indices distinct, so cross that
            // boundary explicitly before forming a scalar operation.
            let (lhs, rhs) = binary_args();
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
            MathOp::ExpFast => kernel.math_approximate(*op, args[0]),
            _ => kernel.math(*op, args[0]),
        },
        PrimitiveId::Select => kernel.select(args[0], args[1], args[2]),
        PrimitiveId::Decode => args[0],
        other => panic!("non-scalar primitive in scalar lowering: {other:?}"),
    }
}

fn lower_constant<B: Backend>(
    kernel: &mut PortableBuilder<'_, B>,
    value: Constant,
    output: &SemanticType,
) -> PortableValue {
    let dtype = match output {
        SemanticType::Scalar(dtype) => *dtype,
        SemanticType::Index { .. } => {
            return match value {
                Constant::Int(v) => kernel
                    .index_constant(u64::try_from(v).expect("checked index constant is negative")),
                _ => panic!("index constant is not integer"),
            };
        }
        _ => panic!("constant output is not scalar"),
    };
    match (value, dtype) {
        (Constant::Bool(v), DType::Bool) => {
            kernel.constant(ConstantValue::Bool(v), ValueType::Bool)
        }
        (Constant::Int(v), DType::I32) => kernel.constant(
            ConstantValue::I32(i32::try_from(v).expect("checked i32 literal is out of range")),
            value_type(dtype),
        ),
        (Constant::Int(v), DType::U32) => kernel.constant(
            ConstantValue::U32(u32::try_from(v).expect("checked u32 literal is out of range")),
            value_type(dtype),
        ),
        (Constant::Float(v), DType::F32) => {
            kernel.constant(ConstantValue::F32(v as f32), value_type(dtype))
        }
        (Constant::Int(v), dtype) if dtype.is_float() => {
            let source = kernel.constant(
                ConstantValue::I32(
                    i32::try_from(v).expect("checked integer literal is out of range"),
                ),
                ValueType::Scalar(DType::I32),
            );
            kernel.cast(source, value_type(dtype))
        }
        (Constant::Float(v), dtype) if dtype.is_float() => {
            let source =
                kernel.constant(ConstantValue::F32(v as f32), ValueType::Scalar(DType::F32));
            kernel.cast(source, value_type(dtype))
        }
        _ => panic!("checked literal/type mismatch"),
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
    expression: AnyExpr,
    values: &SemanticBindings,
) -> CapturedExpr {
    capture_expr_with(arena, expression, &mut |arena, value| {
        let bound = values.get(value);
        let (bound_symbol, dtype, index) = bound.scalar();
        if index {
            CapturedExpr::NatArgument(match arena.symbol_sort(bound_symbol) {
                SymbolSort::Nat => arena.nat_symbol(bound_symbol),
                SymbolSort::Int => {
                    let value = arena.int_symbol(bound_symbol);
                    arena.nat_from_int(value)
                }
                SymbolSort::Scalar(_) => panic!("runtime index has scalar symbol sort"),
            })
        } else {
            CapturedExpr::ScalarArgument(bound_symbol, dtype)
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
fn lower_captured_expr<B: Backend>(
    kernel: &mut PortableBuilder<'_, B>,
    expression: &CapturedExpr,
) -> PortableValue {
    lower_captured_expr_with(kernel, expression, &mut BTreeMap::new())
}
fn lower_captured_expr_with<B: Backend>(
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
                    let zero = zero_of(kernel, r.ty.clone());
                    let more = kernel.cmp(CmpOp::Gt, r, zero);
                    let one = one_of(kernel, q.ty.clone());
                    let zero = zero_of(kernel, q.ty.clone());
                    let increment = kernel.select(more, one, zero);
                    kernel.binary(BinaryOp::Add, q, increment)
                }
                ExprBinary::AlignUp => {
                    let q = kernel.binary(BinaryOp::Div, a, b);
                    let r = kernel.binary(BinaryOp::Rem, a, b);
                    let zero = zero_of(kernel, r.ty.clone());
                    let more = kernel.cmp(CmpOp::Gt, r, zero);
                    let one = one_of(kernel, q.ty.clone());
                    let zero = zero_of(kernel, q.ty.clone());
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
                let constant = match operand.ty {
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
            let result = kernel.repeat(start, end, vec![initial], |kernel, value, carry| {
                binders.insert(*binder, value);
                let term = lower_captured_expr_with(kernel, body, binders);
                binders.remove(binder);
                let result = match op {
                    FoldOp::Sum => kernel.binary(BinaryOp::Add, carry[0], term),
                    FoldOp::Product => kernel.binary(BinaryOp::Mul, carry[0], term),
                    FoldOp::Max => kernel.binary(BinaryOp::Max, carry[0], term),
                };
                vec![result]
            });
            result[0]
        }
    }
}
pub(crate) fn realize_tensor<B: Backend>(
    builder: &mut ImplementationBuilder<'_, B>,
    value: SemanticValueId,
    tensor: TensorSemantics,
) {
    match tensor.storage {
        TensorStorage::Parameter(_) => {
            assert!(
                builder.portable_existing_binding(value).is_some(),
                "checked parameter tensor was not bound to its call argument"
            );
        }
        TensorStorage::Owned => {
            builder.portable_allocate_tensor(value);
        }
        TensorStorage::View { .. } | TensorStorage::Computed => {
            let function = builder.portable_function_ref();
            let mode = if function.values().any(|(_, value)| {
                matches!(
                    value.origin,
                    ValueOrigin::Node(node)
                        if matches!(function.node(node).view(), SemanticNodeView::Intrinsic { .. })
                )
            }) {
                SemanticMode::AuthoredBackend
            } else {
                SemanticMode::Portable
            };
            let mut lowerer = Lowerer::new(function, builder, mode);
            lowerer.realize_pure_value(value, &mut BTreeSet::new());
            let realized = lowerer.values.get(value);
            assert!(
                matches!(realized, Bound::Tensor(_)),
                "materialized semantic value is not a tensor"
            );
        }
    }
}
