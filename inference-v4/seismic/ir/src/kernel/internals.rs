//! Kernel builder and arena internals. `PortableBuilder` is the only kernel
//! constructor.
//!
//! Construction facts established here and never re-checked downstream:
//! - every `ErasedValue` of a closed kernel is dense and typed;
//! - every operand of an op is defined in the op's block or a dominating
//!   block (lexical scoping, §7.2);
//! - every join/carry has one schema (the arms' yielded value types);
//! - every read/write index has the rank of its place;
//! - every declared result slot is written on every path;
//! - every intrinsic's resources and actual selected operation are retained.
//!
//! The only panics are construction bugs against the private builder
//! (a handle of another kernel, a value used outside its scope, a rank
//! mismatch, an unwritten result slot, an unsupported plane) and are
//! §13.3.2 (private arena id) / §13.3.1 (registry) categories.

use super::ops::{
    self, BarrierScope, BinaryOp, Binding, BindingAccess, BitOp, Block, CmpOp, ErasedValue,
    GeometryValue, IntrinsicResources, KernelInterface, LogicOp, MathPrecision,
    Op, PlaceRef, ResourceFacts, UnaryOp, ValueSchema, ValueType,
};
use super::{BindingSlot, BlockId, Kernel, KernelArena, KernelId};
use crate::identity::OwnerToken;
use crate::schedule::AnyScalarSlot;
use crate::storage::{LaunchLocalKind, LocalAllocation};
use crate::target::PhysicalDialect;
use seismic_lang::expr::{ExprArena, NatExpr, SymbolId};
use seismic_lang::ids::{IntrinsicId, RepresentationId};
use seismic_lang::intrinsics::{AtomicOp, MathOp};
use seismic_lang::registry::{
    self, IntrinsicResultType, IntrinsicUniformity, RepresentationKind,
};
use seismic_lang::types::DType;

// ---------------------------------------------------------------------------
// Per-kernel construction state
// ---------------------------------------------------------------------------

struct BlockData<B: PhysicalDialect> {
    ops: Vec<Op<B>>,
    parent: Option<BlockId>,
    /// Product of the trip counts of the enclosing repeats when every one is
    /// an arena expression; `None` when some enclosing trip count is a
    /// runtime scalar with no arena origin.
    multiplicity: Option<NatExpr>,
    /// Uniformity of the complete lexical control path reaching this block.
    control_uniformity: Uniformity,
}

#[derive(Debug)]
struct ValueEntry {
    ty: ValueType,
    block: BlockId,
    /// Arena origin of an index value (constant, nat argument, extent, or
    /// arithmetic over such), used for trip-count derivation of numerical
    /// rounding multiplicities.
    nat: Option<NatExpr>,
    uniformity: Uniformity,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(super) enum Uniformity {
    Workgroup,
    Subgroup,
    Varying,
}

impl Uniformity {
    fn from_intrinsic(value: IntrinsicUniformity) -> Self {
        match value {
            IntrinsicUniformity::Workgroup => Self::Workgroup,
            IntrinsicUniformity::Subgroup => Self::Subgroup,
            IntrinsicUniformity::Varying => Self::Varying,
        }
    }
    fn combine(self, other: Self) -> Self {
        self.max(other)
    }
}

#[derive(Clone, Copy)]
pub(super) struct PlaceEntry {
    place: PlaceRef,
    pub(super) representation: RepresentationId,
    rank: u32,
}

/// The construction state of the one open kernel of an implementation
/// builder. Reset by `close`.
pub(crate) struct KernelState<B: PhysicalDialect> {
    owner: OwnerToken,
    kernel: u32,
    blocks: Vec<BlockData<B>>,
    values: Vec<ValueEntry>,
    places: Vec<PlaceEntry>,
    bindings: Vec<Binding>,
    nat_args: Vec<NatExpr>,
    scalar_args: Vec<(SymbolId, crate::repr::ScalarKind)>,
    result_slots: Vec<AnyScalarSlot>,
    locals: Vec<LocalAllocation>,
    intrinsic_resources: Vec<IntrinsicResources>,
    addressable_resources: Vec<ops::AddressableResourceLease>,
    addressable_resource_cursors: Vec<NatExpr>,
    intrinsics_used: Vec<IntrinsicId>,
    barriers: u32,
    uses_subgroup: bool,
}

impl<B: PhysicalDialect> KernelState<B> {
    pub(crate) fn assert_closed(&self) {
        assert!(
            self.blocks.is_empty(),
            "construction still owns an unfinished kernel"
        );
    }

    pub(crate) fn new(
        owner: OwnerToken,
        kernel: u32,
        zero: NatExpr,
        resource_classes: usize,
    ) -> Self {
        Self {
            owner,
            kernel,
            blocks: Vec::new(),
            values: Vec::new(),
            places: Vec::new(),
            bindings: Vec::new(),
            nat_args: Vec::new(),
            scalar_args: Vec::new(),
            result_slots: Vec::new(),
            locals: Vec::new(),
            intrinsic_resources: Vec::new(),
            addressable_resources: Vec::new(),
            addressable_resource_cursors: vec![zero; resource_classes],
            intrinsics_used: Vec::new(),
            barriers: 0,
            uses_subgroup: false,
        }
    }
}

// ---------------------------------------------------------------------------
// The builder
// ---------------------------------------------------------------------------

pub(crate) struct Builder<'a, B: PhysicalDialect> {
    expr: &'a mut ExprArena,
    storage: &'a crate::storage::TopologyBuilder,
    schedule: &'a crate::schedule::ScheduleConstruction<B>,
    kernels: &'a mut Vec<Kernel<B>>,
    state: &'a mut KernelState<B>,
    target_facts: &'a B::Facts,
    resource_classes: &'a [crate::target::AddressableResourceClass],
    vector_support: &'a crate::target::VectorSupport,
    block: BlockId,
}

/// A value of the source-directed kernel builder.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct PortableValue {
    pub(crate) raw: ErasedValue,
    pub(crate) ty: ValueType,
}
#[derive(Clone, Copy, Debug)]
pub struct PortablePlace {
    owner: OwnerToken,
    kernel: u32,
    index: u32,
    write: PortableWriteCapability,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PortableWriteCapability {
    ReadOnly,
    DenseElement,
    RepresentationPacket,
}

pub type PortableTensor = crate::tensor_view::TensorView<PortablePlace, PortableValue>;
type PortableViewStep = crate::tensor_view::ViewStep<PortableValue>;
pub type PortableSliceAxis = crate::tensor_view::SliceAxis<PortableValue>;

/// Opaque, kernel-owned structured branch under construction. The builder
/// enforces lexical arm order and result schemas when this token is consumed.
#[must_use]
pub struct PortableBranch {
    parent: BlockId,
    then_block: BlockId,
    else_block: BlockId,
    condition: ErasedValue,
    control_uniformity: Uniformity,
}

/// Consuming suspension of the current lexical block. The construction still
/// owns all values, places, operations and control state. This token cannot be
/// copied, minted externally, or used with a different open kernel.
#[derive(Debug)]
pub struct PortableCursor {
    block: BlockId,
}

/// The existing repeat's lexical scope and typed carry schema while its body
/// is open. Closing consumes it; suspended construction keeps the actual block.
pub struct PortableRepeat {
    parent: BlockId,
    body: BlockId,
    start: ErasedValue,
    end: ErasedValue,
    binder: ErasedValue,
    initial: Vec<ErasedValue>,
    parameters: Vec<ErasedValue>,
    schema: ValueSchema,
    recurrence: Vec<Uniformity>,
}

pub struct PortableBuilder<'a, B: PhysicalDialect> {
    inner: Builder<'a, B>,
}

/// Authority to bind the logical base of a one-dimensional semantic launch.
/// Only the kernel operation constructor can issue this handle.
///
/// ```compile_fail
/// use seismic_ir::kernel::dynamic::LogicalIndexBinding;
/// fn raw_ordinal_is_not_a_binding() -> LogicalIndexBinding { 0u32 }
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct LogicalIndexBinding {
    pub(crate) kernel: KernelId,
    pub(crate) argument: u32,
    pub(crate) extent: NatExpr,
}

pub(crate) fn open_portable<'a, B: PhysicalDialect>(
    owner: OwnerToken,
    expr: &'a mut ExprArena,
    storage: &'a crate::storage::TopologyBuilder,
    schedule: &'a crate::schedule::ScheduleConstruction<B>,
    kernels: &'a mut Vec<Kernel<B>>,
    state: &'a mut KernelState<B>,
    target_facts: &'a B::Facts,
    resource_classes: &'a [crate::target::AddressableResourceClass],
    vector_support: &'a crate::target::VectorSupport,
) -> PortableBuilder<'a, B> {
    state.assert_closed();
    let kernel = kernels.len() as u32;
    let zero = expr.nat(0);
    let one = expr.nat(1);
    *state = KernelState::new(owner, kernel, zero, resource_classes.len());
    state.blocks.push(BlockData {
        ops: Vec::new(),
        parent: None,
        multiplicity: Some(one),
        control_uniformity: Uniformity::Workgroup,
    });
    PortableBuilder {
        inner: Builder {
            expr,
            storage,
            schedule,
            kernels,
            state,
            target_facts,
            resource_classes,
            vector_support,
            block: BlockId::new(owner, kernel, 0),
        },
    }
}

pub(crate) fn resume_portable<'a, B: PhysicalDialect>(
    owner: OwnerToken,
    expr: &'a mut ExprArena,
    storage: &'a crate::storage::TopologyBuilder,
    schedule: &'a crate::schedule::ScheduleConstruction<B>,
    kernels: &'a mut Vec<Kernel<B>>,
    state: &'a mut KernelState<B>,
    target_facts: &'a B::Facts,
    resource_classes: &'a [crate::target::AddressableResourceClass],
    vector_support: &'a crate::target::VectorSupport,
    cursor: PortableCursor,
) -> PortableBuilder<'a, B> {
    assert_eq!(
        cursor.block.owner(),
        owner,
        "kernel cursor belongs to another construction"
    );
    assert_eq!(
        cursor.block.kernel(),
        state.kernel,
        "kernel cursor belongs to another open kernel"
    );
    assert_eq!(
        state.kernel as usize,
        kernels.len(),
        "kernel cursor refers to a closed kernel"
    );
    assert!(
        (cursor.block.index() as usize) < state.blocks.len(),
        "kernel cursor has no open lexical block"
    );
    PortableBuilder {
        inner: Builder {
            expr,
            storage,
            schedule,
            kernels,
            state,
            target_facts,
            resource_classes,
            vector_support,
            block: cursor.block,
        },
    }
}

fn dtype_value_type(dtype: DType) -> ValueType {
    if dtype == DType::Bool {
        ValueType::Bool
    } else {
        ValueType::Scalar(dtype)
    }
}

impl<'a, B: PhysicalDialect> PortableBuilder<'a, B> {
    /// End the temporary borrow while retaining the actual current block in
    /// its construction. Resume through that same construction before close.
    pub fn suspend(self) -> PortableCursor {
        PortableCursor {
            block: self.inner.block,
        }
    }

    fn checked_place(&self, place: PortablePlace) -> PlaceEntry {
        self.inner.assert_kernel(place.owner, place.kernel);
        *self
            .inner
            .state
            .places
            .get(place.index as usize)
            .expect("place is outside its kernel")
    }

    /// Derives operand metadata from an owned value instead of trusting a
    /// caller-supplied uniformity or index flag.
    pub fn semantic_scalar(&mut self, value: PortableValue, dtype: DType) -> ops::SemanticScalar {
        self.inner.use_value(value.raw, &value.ty);
        assert!(
            value.ty == dtype_value_type(dtype)
                || (value.ty == ValueType::Index && dtype == DType::U32),
            "intrinsic scalar dtype mismatch"
        );
        ops::SemanticScalar {
            value,
            dtype,
            index: value.ty == ValueType::Index,
            uniformity: self.uniformity(value),
        }
    }
    /// Rebind a capability value after an ordinary same-type SSA join.
    pub fn opaque_with_value(
        &mut self,
        original: ops::SemanticOpaque,
        value: PortableValue,
    ) -> ops::SemanticOpaque {
        assert_eq!(
            original.value.ty, value.ty,
            "opaque transport changed its type"
        );
        self.used(value);
        ops::SemanticOpaque {
            value,
            uniformity: self.uniformity(value),
            ..original
        }
    }
    pub fn semantic_place(
        &self,
        tensor: PortableTensor,
        representation: RepresentationId,
        rank: u32,
        writable: bool,
    ) -> ops::SemanticPlace {
        assert_eq!(
            self.tensor_mapping(&tensor).representation,
            representation,
            "intrinsic place representation mismatch"
        );
        assert_eq!(
            tensor.extents.len(),
            rank as usize,
            "intrinsic place rank mismatch"
        );
        assert!(
            !writable
                || (!tensor.has_plane() && tensor.place.write != PortableWriteCapability::ReadOnly),
            "read-only intrinsic place cannot be writable"
        );
        ops::SemanticPlace {
            tensor,
            representation,
            rank,
            writable,
        }
    }

    /// Exact host expression retained by this actual value's constructor.
    pub fn exact_nat(&self, value: PortableValue) -> Option<NatExpr> {
        self.inner.nat_of(value.raw)
    }

    /// Geometry of the actual schedule-owned tensor argument.
    pub fn view_extents(&self, view: crate::storage::AnyBufferView) -> &[NatExpr] {
        assert_eq!(view.owner(), self.inner.owner(), "foreign tensor contract");
        &self.inner.storage.view_layout(view).extents
    }

    pub fn expression_arena(&mut self) -> &mut ExprArena {
        self.inner.expr
    }

    pub fn arg_view(
        &mut self,
        view: crate::storage::AnyBufferView,
        writable: bool,
    ) -> PortablePlace {
        if writable
            && !matches!(
                registry::representation_info(view.representation).kind,
                RepresentationKind::Dense(_)
            )
        {
            panic!(
                "portable lowering cannot form a general write to a non-writable representation"
            );
        }
        let access = if writable {
            BindingAccess::Write
        } else {
            BindingAccess::Read
        };
        let (slot, rank) = self.inner.bind_view(view, access);
        let index = self
            .inner
            .push_place(PlaceRef::Global { slot }, view.representation, rank);
        PortablePlace {
            owner: self.inner.owner(),
            kernel: self.inner.state.kernel,
            index,
            write: if writable {
                PortableWriteCapability::DenseElement
            } else {
                PortableWriteCapability::ReadOnly
            },
        }
    }
    pub fn representation_destination(
        &mut self,
        view: crate::storage::AnyBufferView,
    ) -> PortablePlace {
        assert!(
            matches!(
                registry::representation_info(view.representation).kind,
                RepresentationKind::Packed(_)
            ),
            "representation conversion destination must use packed storage"
        );
        let (slot, rank) = self.inner.bind_view(view, BindingAccess::Write);
        let index = self
            .inner
            .push_place(PlaceRef::Global { slot }, view.representation, rank);
        PortablePlace {
            owner: self.inner.owner(),
            kernel: self.inner.state.kernel,
            index,
            write: PortableWriteCapability::RepresentationPacket,
        }
    }
    pub fn local_tensor(
        &mut self,
        kind: LaunchLocalKind,
        representation: RepresentationId,
        extents: Vec<NatExpr>,
    ) -> PortableTensor {
        let info = registry::representation_info(representation);
        assert_eq!(
            info.access,
            registry::RepresentationAccess::ReadWrite,
            "portable local tensors require a writable registered representation"
        );
        let local = self.inner.state.locals.len() as u32;
        self.inner.state.locals.push(LocalAllocation {
            kind,
            representation,
            extents,
            alignment: representation_alignment(representation),
        });
        let rank = self.inner.state.locals[local as usize].extents.len() as u32;
        let index = self
            .inner
            .push_place(PlaceRef::Local { index: local }, representation, rank);
        self.tensor(PortablePlace {
            owner: self.inner.owner(),
            kernel: self.inner.state.kernel,
            index,
            write: PortableWriteCapability::DenseElement,
        })
    }
    pub fn nat_arg(&mut self, expr: NatExpr) -> PortableValue {
        self.nat_arg_with_ordinal(expr).0
    }
    /// Emits the logical coordinate and issues its matching launch binding.
    /// Padded participants use the representable extent sentinel, avoiding an
    /// overflowing addition before the caller's active-participant test.
    pub fn logical_global_id(&mut self, extent: NatExpr) -> (PortableValue, LogicalIndexBinding) {
        let zero = self.inner.expr.nat(0);
        let (base, argument) = self.nat_arg_with_ordinal(zero);
        // The launch supplies this argument after chunk normalization. Its
        // nominal ABI default is not an exact kernel-owned expression.
        self.inner.state.values[base.raw.index() as usize].nat = None;
        let end = self.nat_arg(extent);
        let remaining = self.binary(BinaryOp::Sub, end, base);
        let physical = self.global_id(0);
        let offset = self.binary(BinaryOp::Min, physical, remaining);
        let logical = self.binary(BinaryOp::Add, base, offset);
        (
            logical,
            LogicalIndexBinding {
                kernel: KernelId::new(self.inner.owner(), self.inner.state.kernel),
                argument,
                extent,
            },
        )
    }

    /// Read this kernel's existing logical-base argument. Cohort membership
    /// uses the same launch binding as participant coordinates.
    pub fn logical_base(&mut self, binding: LogicalIndexBinding) -> PortableValue {
        assert_eq!(
            binding.kernel,
            KernelId::new(self.inner.owner(), self.inner.state.kernel)
        );
        let out = self
            .inner
            .define_with(ValueType::Index, None, Uniformity::Workgroup);
        self.inner.emit(Op::NatArg {
            out,
            index: binding.argument,
        });
        PortableValue {
            raw: out,
            ty: ValueType::Index,
        }
    }
    fn nat_arg_with_ordinal(&mut self, expr: NatExpr) -> (PortableValue, u32) {
        let index = self.inner.state.nat_args.len() as u32;
        self.inner.state.nat_args.push(expr);
        let out = self
            .inner
            .define_with(ValueType::Index, Some(expr), Uniformity::Workgroup);
        self.inner.emit(Op::NatArg { out, index });
        (
            PortableValue {
                raw: out,
                ty: ValueType::Index,
            },
            index,
        )
    }
    pub fn scalar_arg(&mut self, symbol: SymbolId, dtype: DType) -> PortableValue {
        let index = self.inner.state.scalar_args.len() as u32;
        self.inner
            .state
            .scalar_args
            .push((symbol, crate::repr::ScalarKind::Scalar(dtype)));
        let ty = dtype_value_type(dtype);
        let out = self
            .inner
            .define_with(ty.clone(), None, Uniformity::Workgroup);
        self.inner.emit(Op::ScalarArg { out, index });
        PortableValue { raw: out, ty }
    }
    fn geometry(&mut self, kind: GeometryValue) -> PortableValue {
        PortableValue {
            raw: self.inner.geometry(kind),
            ty: ValueType::Index,
        }
    }
    pub fn global_id(&mut self, axis: u8) -> PortableValue {
        self.geometry(GeometryValue::GlobalId(axis))
    }
    pub fn local_id(&mut self, axis: u8) -> PortableValue {
        self.geometry(GeometryValue::LocalId(axis))
    }
    pub fn workgroup_id(&mut self, axis: u8) -> PortableValue {
        self.geometry(GeometryValue::WorkgroupId(axis))
    }
    pub fn workgroup_size(&mut self, axis: u8) -> PortableValue {
        self.geometry(GeometryValue::WorkgroupSize(axis))
    }
    pub fn subgroup_lane(&mut self) -> PortableValue {
        self.inner.state.uses_subgroup = true;
        self.geometry(GeometryValue::SubgroupLane)
    }
    /// Stable subgroup position within the current workgroup.
    pub fn subgroup_ordinal(&mut self) -> PortableValue {
        self.inner.state.uses_subgroup = true;
        self.geometry(GeometryValue::SubgroupOrdinal)
    }
    /// Actual native subgroup width for this compiled kernel.
    pub fn subgroup_size(&mut self) -> PortableValue {
        self.inner.state.uses_subgroup = true;
        self.geometry(GeometryValue::SubgroupSize)
    }
    pub fn workgroup_barrier(&mut self) {
        self.inner.barrier(ops::BarrierScope::Workgroup)
    }
    pub fn subgroup_barrier(&mut self) {
        self.inner.barrier(ops::BarrierScope::Subgroup);
    }

    /// Rebind a real view product after a structured SSA join. Every supplied
    /// field must dominate the current block and retain its original type.
    pub fn tensor_with_scalar_fields(
        &mut self,
        tensor: &PortableTensor,
        fields: &[PortableValue],
    ) -> PortableTensor {
        self.checked_place(tensor.place);
        let original = tensor.scalar_fields();
        assert_eq!(
            original.len(),
            fields.len(),
            "tensor scalar field arity changed"
        );
        for (before, after) in original.iter().zip(fields) {
            assert_eq!(before.ty, after.ty, "tensor scalar field type changed");
            self.used(*after);
        }
        let mut fields = fields.iter().copied();
        let out = tensor.map(|place| *place, |_| fields.next().unwrap());
        assert!(fields.next().is_none());
        out
    }

    pub fn uniformity(&self, value: PortableValue) -> seismic_lang::registry::IntrinsicUniformity {
        match self.inner.uniformity_of(value.raw) {
            Uniformity::Workgroup => seismic_lang::registry::IntrinsicUniformity::Workgroup,
            Uniformity::Subgroup => seismic_lang::registry::IntrinsicUniformity::Subgroup,
            Uniformity::Varying => seismic_lang::registry::IntrinsicUniformity::Varying,
        }
    }
    pub fn result_slot(&mut self, slot: AnyScalarSlot) -> u32 {
        assert_eq!(
            slot.owner(),
            self.inner.owner(),
            "result slot belongs to another implementation"
        );
        let index = self.inner.state.result_slots.len() as u32;
        self.inner.state.result_slots.push(slot);
        index
    }
    pub fn index_constant(&mut self, value: u64) -> PortableValue {
        let nat = self.inner.expr.nat(value);
        let out = self
            .inner
            .define_with(ValueType::Index, Some(nat), Uniformity::Workgroup);
        self.inner.emit(Op::Constant {
            out,
            value: ops::ConstantValue::Index(value),
        });
        PortableValue {
            raw: out,
            ty: ValueType::Index,
        }
    }
    pub fn constant(&mut self, value: ops::ConstantValue, ty: ValueType) -> PortableValue {
        let value = match (value, &ty) {
            (value @ ops::ConstantValue::F32(_), ValueType::Scalar(DType::F32))
            | (value @ ops::ConstantValue::F16(_), ValueType::Scalar(DType::F16))
            | (value @ ops::ConstantValue::BF16(_), ValueType::Scalar(DType::BF16))
            | (value @ ops::ConstantValue::I32(_), ValueType::Scalar(DType::I32))
            | (value @ ops::ConstantValue::U32(_), ValueType::Scalar(DType::U32))
            | (value @ ops::ConstantValue::Bool(_), ValueType::Bool)
            | (value @ ops::ConstantValue::Index(_), ValueType::Index) => value,
            _ => panic!("portable constant payload differs from its closed value type"),
        };
        let out = self
            .inner
            .define_with(ty.clone(), None, Uniformity::Workgroup);
        self.inner.emit(Op::Constant { out, value });
        PortableValue { raw: out, ty }
    }
    fn used(&mut self, value: PortableValue) -> ErasedValue {
        self.inner.use_value(value.raw, &value.ty)
    }
    /// Construct source arithmetic and consume its failures at the current
    /// continuation before exposing the value and successful liveness. The
    /// destination resolver only routes typed causes to existing status storage.
    pub fn source_scalar(
        &mut self,
        operation: seismic_lang::reference_math::ScalarOp,
        operands: &[PortableValue],
        alive: PortableValue,
        destination: impl FnMut(seismic_lang::reference_math::ScalarFailure) -> (PortableTensor, PortableValue),
    ) -> (PortableValue, PortableValue) {
        super::reference_math::continue_scalar(self, operation, operands, alive, destination)
    }
    pub fn binary(&mut self, op: BinaryOp, a: PortableValue, b: PortableValue) -> PortableValue {
        if a.ty == ValueType::Index {
            return self.binary_terminal(op, a, b);
        }
        use seismic_lang::{reference_math::ScalarOp, syntax::ast::BinaryOp as Source};
        let op = match op {
            BinaryOp::Add => ScalarOp::Binary(Source::Add),
            BinaryOp::Sub => ScalarOp::Binary(Source::Sub),
            BinaryOp::Mul => ScalarOp::Binary(Source::Mul),
            BinaryOp::Div => ScalarOp::Binary(Source::Div),
            BinaryOp::Rem => ScalarOp::Binary(Source::Rem),
            BinaryOp::Min => ScalarOp::Math(MathOp::Min),
            BinaryOp::Max => ScalarOp::Math(MathOp::Max),
        };
        super::reference_math::expand_total(self, op, &[a, b])
    }
    pub fn bit(&mut self, op: BitOp, a: PortableValue, b: PortableValue) -> PortableValue {
        if a.ty == ValueType::Index {
            return self.bit_terminal(op, a, b);
        }
        use seismic_lang::{reference_math::ScalarOp, syntax::ast::BinaryOp as Source};
        let op = match op {
            BitOp::And => Source::BitAnd,
            BitOp::Or => Source::BitOr,
            BitOp::Xor => Source::BitXor,
            BitOp::Shl => Source::Shl,
            BitOp::Shr => Source::Shr,
        };
        super::reference_math::expand_total(self, ScalarOp::Binary(op), &[a, b])
    }
    pub fn cmp(&mut self, op: CmpOp, a: PortableValue, b: PortableValue) -> PortableValue {
        if a.ty == ValueType::Index {
            return self.cmp_terminal(op, a, b);
        }
        use seismic_lang::{reference_math::ScalarOp, syntax::ast::BinaryOp as Source};
        let op = match op {
            CmpOp::Eq => Source::Eq,
            CmpOp::Ne => Source::Ne,
            CmpOp::Lt => Source::Lt,
            CmpOp::Le => Source::Le,
            CmpOp::Gt => Source::Gt,
            CmpOp::Ge => Source::Ge,
        };
        super::reference_math::expand_total(self, ScalarOp::Binary(op), &[a, b])
    }
    pub(super) fn binary_terminal(
        &mut self,
        op: BinaryOp,
        a: PortableValue,
        b: PortableValue,
    ) -> PortableValue {
        assert_eq!(a.ty, b.ty, "portable binary operand types differ");
        assert_ne!(
            a.ty,
            ValueType::Bool,
            "boolean arithmetic is not constructible"
        );
        let (a, b) = (self.used(a), self.used(b));
        let ty = a_type(&self.inner, a);
        let out = self.inner.binary_value(op, ty, a, b);
        PortableValue { raw: out, ty }
    }
    pub fn unary(&mut self, op: UnaryOp, a: PortableValue) -> PortableValue {
        use seismic_lang::{reference_math::ScalarOp, syntax::ast};
        let op = match op {
            UnaryOp::Neg => ScalarOp::Unary(ast::UnaryOp::Neg),
            UnaryOp::Abs => ScalarOp::Math(MathOp::Abs),
        };
        super::reference_math::expand_total(self, op, &[a])
    }
    pub(super) fn bit_terminal(
        &mut self,
        op: BitOp,
        a: PortableValue,
        b: PortableValue,
    ) -> PortableValue {
        assert_eq!(a.ty, b.ty, "portable bit operand types differ");
        let ty = a.ty.clone();
        let uniformity = self.inner.combined_uniformity([a.raw, b.raw]);
        let (a, b) = (self.used(a), self.used(b));
        let out = self.inner.define_with(ty.clone(), None, uniformity);
        self.inner.emit(Op::Bit { op, out, a, b });
        PortableValue { raw: out, ty }
    }
    pub(super) fn cmp_terminal(
        &mut self,
        op: CmpOp,
        a: PortableValue,
        b: PortableValue,
    ) -> PortableValue {
        assert_eq!(a.ty, b.ty, "portable comparison operand types differ");
        let uniformity = self.inner.combined_uniformity([a.raw, b.raw]);
        let (a, b) = (self.used(a), self.used(b));
        let out = self.inner.define_with(ValueType::Bool, None, uniformity);
        self.inner.emit(Op::Cmp { op, out, a, b });
        PortableValue {
            raw: out,
            ty: ValueType::Bool,
        }
    }
    pub fn logic(&mut self, op: LogicOp, a: PortableValue, b: PortableValue) -> PortableValue {
        assert_eq!(a.ty, ValueType::Bool);
        assert_eq!(b.ty, ValueType::Bool);
        let uniformity = self.inner.combined_uniformity([a.raw, b.raw]);
        let (a, b) = (self.used(a), self.used(b));
        let out = self.inner.define_with(ValueType::Bool, None, uniformity);
        self.inner.emit(Op::Logic { op, out, a, b });
        PortableValue {
            raw: out,
            ty: ValueType::Bool,
        }
    }
    pub fn select(
        &mut self,
        condition: PortableValue,
        a: PortableValue,
        b: PortableValue,
    ) -> PortableValue {
        assert_eq!(condition.ty, ValueType::Bool);
        assert_eq!(a.ty, b.ty, "portable select operand types differ");
        let ty = a.ty.clone();
        let uniformity = self
            .inner
            .combined_uniformity([condition.raw, a.raw, b.raw]);
        let condition = self.used(condition);
        let (a, b) = (self.used(a), self.used(b));
        let out = self.inner.define_with(ty.clone(), None, uniformity);
        self.inner.emit(Op::Select {
            out,
            cond: condition,
            a,
            b,
        });
        PortableValue { raw: out, ty }
    }
    pub fn not(&mut self, a: PortableValue) -> PortableValue {
        assert_eq!(a.ty, ValueType::Bool);
        let uniformity = self.inner.uniformity_of(a.raw);
        let a = self.used(a);
        let out = self.inner.define_with(ValueType::Bool, None, uniformity);
        self.inner.emit(Op::Not { out, a });
        PortableValue {
            raw: out,
            ty: ValueType::Bool,
        }
    }
    pub fn cast(&mut self, a: PortableValue, to: ValueType) -> PortableValue {
        if matches!(a.ty, ValueType::Scalar(_) | ValueType::Bool)
            && matches!(to, ValueType::Scalar(_) | ValueType::Bool)
        {
            let dtype = match to {
                ValueType::Scalar(dtype) => dtype,
                ValueType::Bool => DType::Bool,
                _ => unreachable!(),
            };
            return super::reference_math::expand_total(
                self,
                seismic_lang::reference_math::ScalarOp::Cast(dtype),
                &[a],
            );
        }
        let uniformity = self.inner.uniformity_of(a.raw);
        let a = self.used(a);
        let out = self.inner.define_with(to.clone(), None, uniformity);
        self.inner.emit(Op::Cast {
            out,
            a,
            to: to.clone(),
        });
        PortableValue { raw: out, ty: to }
    }
    pub fn bitcast(&mut self, a: PortableValue, to: ValueType) -> PortableValue {
        // Bitcast transports a complete scalar payload. Equal byte counts do
        // not define vector lane packing or make Boolean carriers bit fields.
        // The scalar relation and all native emitters agree on these two
        // numeric payload widths; other reshaping needs an explicit operation.
        let supported = match (a.ty, to) {
            (ValueType::Scalar(from), ValueType::Scalar(into))
                if from != DType::Bool && into != DType::Bool =>
            {
                from.bytes() == into.bytes()
            }
            _ => false,
        };
        assert!(
            supported,
            "bitcast requires equal-width numeric scalar payloads"
        );
        let uniformity = self.inner.uniformity_of(a.raw);
        let a = self.used(a);
        let out = self.inner.define_with(to.clone(), None, uniformity);
        self.inner.emit(Op::Bitcast {
            out,
            a,
            to: to.clone(),
        });
        PortableValue { raw: out, ty: to }
    }
    /// Payload transport: scalar representation bits, never a numeric cast.
    /// Only the genuinely narrower payload needs its own operation; equal-width
    /// and canonical Boolean cases use the existing ordinary IR operations.
    pub fn scalar_bits(&mut self, a: PortableValue) -> PortableValue {
        let to = ValueType::Scalar(DType::U32);
        match a.ty {
            ValueType::Scalar(DType::U32) => a,
            ValueType::Scalar(DType::F32 | DType::I32) => self.bitcast(a, to),
            ValueType::Bool => {
                let one = self.constant(ops::ConstantValue::U32(1), to);
                let zero = self.constant(ops::ConstantValue::U32(0), to);
                self.select(a, one, zero)
            }
            ValueType::Scalar(DType::F16 | DType::BF16) => {
                let uniformity = self.inner.uniformity_of(a.raw);
                let a = self.used(a);
                let out = self.inner.define_with(to, None, uniformity);
                self.inner.emit(Op::ScalarBits { out, a });
                PortableValue { raw: out, ty: to }
            }
            _ => panic!("scalar payload transport requires a source scalar"),
        }
    }
    pub fn scalar_from_bits(&mut self, a: PortableValue, dtype: DType) -> PortableValue {
        assert_eq!(a.ty, ValueType::Scalar(DType::U32));
        let to = if dtype == DType::Bool {
            ValueType::Bool
        } else {
            ValueType::Scalar(dtype)
        };
        match dtype {
            DType::U32 => a,
            DType::F32 | DType::I32 => self.bitcast(a, to),
            DType::Bool => {
                let zero = self.constant(ops::ConstantValue::U32(0), a.ty);
                self.cmp_terminal(CmpOp::Ne, a, zero)
            }
            DType::F16 | DType::BF16 => {
                let uniformity = self.inner.uniformity_of(a.raw);
                let a = self.used(a);
                let out = self.inner.define_with(to, None, uniformity);
                self.inner.emit(Op::ScalarFromBits { out, a });
                PortableValue { raw: out, ty: to }
            }
        }
    }
    pub fn math(&mut self, op: MathOp, a: PortableValue) -> PortableValue {
        super::reference_math::expand(self, op, a)
    }
    pub fn math_approximate(&mut self, op: MathOp, a: PortableValue) -> PortableValue {
        let ty = a.ty;
        let uniformity = self.inner.uniformity_of(a.raw);
        let a = self.used(a);
        let out = self.inner.define_with(ty, None, uniformity);
        self.inner.emit(Op::Math {
            op,
            precision: MathPrecision::Approximate,
            out,
            a,
        });
        PortableValue { raw: out, ty }
    }
    pub fn fma(&mut self, a: PortableValue, b: PortableValue, c: PortableValue) -> PortableValue {
        super::reference_math::expand_total(
            self,
            seismic_lang::reference_math::ScalarOp::Math(MathOp::Fma),
            &[a, b, c],
        )
    }

    pub fn read(&mut self, place: PortablePlace, index: &[PortableValue]) -> PortableValue {
        let entry = self.checked_place(place);
        self.read_entry(entry, index)
    }
    fn read_entry(&mut self, entry: PlaceEntry, index: &[PortableValue]) -> PortableValue {
        match &registry::representation_info(entry.representation).kind {
            RepresentationKind::Packed(_) => super::representation::read(self, entry, index),
            RepresentationKind::Dense(dtype) => {
                let indices = self.read_indices(entry, index);
                let ty = dtype_value_type(*dtype);
                let uniformity = self
                    .inner
                    .read_uniformity(entry.place, indices.iter().copied());
                let out = self.inner.define_with(ty, None, uniformity);
                self.inner.emit(Op::Read {
                    out,
                    place: entry.place,
                    representation: entry.representation,
                    index: indices,
                });
                PortableValue { raw: out, ty }
            }
            RepresentationKind::External(_) => {
                panic!("external packets require a registered conversion")
            }
        }
    }
    fn read_indices(&mut self, entry: PlaceEntry, index: &[PortableValue]) -> Vec<ErasedValue> {
        assert_eq!(
            entry.rank as usize,
            index.len(),
            "portable read rank mismatch"
        );
        index
            .iter()
            .map(|value| {
                assert_eq!(value.ty, ValueType::Index);
                self.inner.use_value(value.raw, &ValueType::Index)
            })
            .collect()
    }
    pub(super) fn read_plane_field(
        &mut self,
        entry: PlaceEntry,
        index: &[PortableValue],
        plane: u32,
        field: u32,
    ) -> PortableValue {
        let RepresentationKind::Packed(layout) =
            &registry::representation_info(entry.representation).kind
        else {
            panic!("field read requires a packed representation")
        };
        let schema = &layout.planes[plane as usize];
        assert!(field < schema.fields, "field is outside its plane group");
        let dtype = match schema.encoding {
            registry::PlaneEncoding::Dense(dtype) => dtype,
            registry::PlaneEncoding::Packed { .. } | registry::PlaneEncoding::FloatCode { .. } => {
                DType::U32
            }
        };
        let indices = self.read_indices(entry, index);
        let ty = dtype_value_type(dtype);
        let uniformity = self
            .inner
            .read_uniformity(entry.place, indices.iter().copied());
        let out = self.inner.define_with(ty, None, uniformity);
        self.inner.emit(Op::ReadPlaneField {
            out,
            place: entry.place,
            plane,
            field,
            index: indices,
        });
        PortableValue { raw: out, ty }
    }
    pub fn write(&mut self, place: PortablePlace, index: &[PortableValue], value: PortableValue) {
        assert_eq!(
            place.write,
            PortableWriteCapability::DenseElement,
            "portable write requires dense-element write authority"
        );
        let entry = self.checked_place(place);
        let dtype = match registry::representation_info(entry.representation).kind {
            RepresentationKind::Dense(dtype) => dtype,
            RepresentationKind::Packed(_) => {
                panic!("packed writes have no canonical encode contract")
            }
            RepresentationKind::External(_) => {
                panic!("external packets are never general writable elements")
            }
        };
        assert_eq!(
            value.ty,
            dtype_value_type(dtype),
            "portable write element type mismatch"
        );
        assert_eq!(
            entry.rank as usize,
            index.len(),
            "portable write rank mismatch"
        );
        let indices = index
            .iter()
            .map(|value| {
                assert_eq!(value.ty, ValueType::Index);
                self.inner.use_value(value.raw, &ValueType::Index)
            })
            .collect();
        let value = self.used(value);
        self.inner.emit(Op::Write {
            place: entry.place,
            representation: entry.representation,
            index: indices,
            value,
        });
    }
    pub fn atomic(
        &mut self,
        op: AtomicOp,
        place: PortablePlace,
        index: &[PortableValue],
        value: PortableValue,
    ) {
        assert_eq!(
            place.write,
            PortableWriteCapability::DenseElement,
            "portable atomic requires dense-element write authority"
        );
        let entry = self.checked_place(place);
        let dtype = match registry::representation_info(entry.representation).kind {
            RepresentationKind::Dense(dtype) => dtype,
            RepresentationKind::Packed(_) => {
                panic!("packed atomics have no canonical update contract")
            }
            RepresentationKind::External(_) => {
                panic!("external packets have no atomic element contract")
            }
        };
        assert!(
            matches!(
                dtype,
                DType::F32 | DType::F16 | DType::BF16 | DType::I32 | DType::U32
            ),
            "atomic dtype is not registered"
        );
        assert_eq!(value.ty, dtype_value_type(dtype));
        assert_eq!(entry.rank as usize, index.len());
        let indices = index
            .iter()
            .map(|item| {
                assert_eq!(item.ty, ValueType::Index);
                self.inner.use_value(item.raw, &ValueType::Index)
            })
            .collect();
        let value = self.used(value);
        self.inner.emit(Op::Atomic {
            op,
            place: entry.place,
            representation: entry.representation,
            index: indices,
            value,
        });
    }
    pub fn extent(&mut self, place: PortablePlace, axis: u32) -> PortableValue {
        let entry = self.checked_place(place);
        assert!(
            axis < entry.rank,
            "portable extent axis is outside the place rank"
        );
        let nat = match entry.place {
            PlaceRef::Global { slot } => {
                let view = self.inner.state.bindings[slot.index() as usize].view;
                Some(self.inner.storage.views()[view.index as usize].extents[axis as usize])
            }
            PlaceRef::Local { index } => {
                Some(self.inner.state.locals[index as usize].extents[axis as usize])
            }
        };
        let out = self
            .inner
            .define_with(ValueType::Index, nat, Uniformity::Workgroup);
        self.inner.emit(Op::Extent {
            out,
            place: entry.place,
            axis,
        });
        PortableValue {
            raw: out,
            ty: ValueType::Index,
        }
    }
    pub fn tensor(&mut self, place: PortablePlace) -> PortableTensor {
        let rank = self.checked_place(place).rank;
        let extents = (0..rank).map(|axis| self.extent(place, axis)).collect();
        PortableTensor {
            place,
            extents,
            steps: Vec::new(),
        }
    }
    /// Query the existing storage and region-product owners. A logical view
    /// keeps the identity of its backing; distinct participant locals are
    /// disjoint, while external argument aliases follow the entry contract.
    pub fn tensors_may_overlap(&self, a: &PortableTensor, b: &PortableTensor) -> bool {
        match (self.checked_place(a.place).place, self.checked_place(b.place).place) {
            (PlaceRef::Local { index: a }, PlaceRef::Local { index: b }) => a == b,
            (PlaceRef::Local { .. }, PlaceRef::Global { .. })
            | (PlaceRef::Global { .. }, PlaceRef::Local { .. }) => false,
            (PlaceRef::Global { slot: a }, PlaceRef::Global { slot: b }) => {
                let a = self.inner.state.bindings[a.index() as usize].view;
                let b = self.inner.state.bindings[b.index() as usize].view;
                self.inner.storage.may_overlap_views(self.inner.schedule, a, b)
            }
        }
    }

    pub fn tensor_extents<'b>(&self, tensor: &'b PortableTensor) -> &'b [PortableValue] {
        &tensor.extents
    }
    pub fn representation_convert_packet(
        &mut self,
        source: &PortableTensor,
        destination: &PortableTensor,
        conversion: seismic_lang::ids::RepresentationConversionId,
        packet: PortableValue,
    ) {
        let recipe = registry::representation_conversion_info(conversion);
        let source_place = self.checked_place(source.place);
        let destination_place = self.checked_place(destination.place);
        assert_eq!(
            source_place.representation, recipe.source,
            "conversion source representation differs from its registered recipe"
        );
        assert_eq!(
            destination_place.representation, recipe.destination,
            "conversion destination representation differs from its registered recipe"
        );
        assert_eq!(
            destination.place.write,
            PortableWriteCapability::RepresentationPacket,
            "representation conversion requires packet-write authority"
        );
        assert!(
            source.steps.is_empty() && !source.has_plane(),
            "external conversion source must be the canonical complete packet view"
        );
        assert!(
            destination.steps.is_empty() && !destination.has_plane(),
            "resident conversion destination must be the canonical complete packet view"
        );
        assert_eq!(
            packet.ty,
            ValueType::Index,
            "conversion packet ordinal must be an index"
        );
        let packet = self.used(packet);
        let PlaceRef::Global { slot: source } = source_place.place else {
            panic!("external conversion source must be a global binding")
        };
        let PlaceRef::Global { slot: destination } = destination_place.place else {
            panic!("resident conversion destination must be a global binding")
        };
        self.inner.emit(Op::RepresentationConvertPacket {
            source,
            destination,
            conversion,
            packet,
        });
    }
    fn tensor_mapping(&self, tensor: &PortableTensor) -> ops::LogicalTensorMap {
        let entry = self.checked_place(tensor.place);
        let steps = tensor
            .steps
            .iter()
            .map(|step| match step {
                PortableViewStep::Plane { plane, axis } => ops::LogicalViewStep::Plane {
                    plane: *plane,
                    axis: *axis as u32,
                },
                PortableViewStep::Slice(axes) => ops::LogicalViewStep::Slice(
                    axes.iter()
                        .map(|axis| match axis {
                            PortableSliceAxis::Point(value) => {
                                ops::LogicalSliceAxis::Point(value.raw)
                            }
                            PortableSliceAxis::Range { start, end } => {
                                ops::LogicalSliceAxis::Range {
                                    start: start.raw,
                                    end: end.raw,
                                }
                            }
                            PortableSliceAxis::Full => ops::LogicalSliceAxis::Full,
                        })
                        .collect(),
                ),
                PortableViewStep::Transpose(permutation) => {
                    ops::LogicalViewStep::Transpose(permutation.clone())
                }
                PortableViewStep::Reshape { from, to } => ops::LogicalViewStep::Reshape {
                    from: from.iter().map(|value| value.raw).collect(),
                    to: to.iter().map(|value| value.raw).collect(),
                },
            })
            .collect();
        let representation = tensor
            .steps
            .iter()
            .find_map(|step| match step {
                PortableViewStep::Plane { plane, .. } => {
                    let RepresentationKind::Packed(layout) =
                        &registry::representation_info(entry.representation).kind
                    else {
                        panic!("plane projection requires packed backing");
                    };
                    Some(registry::dense(
                        layout.planes[*plane as usize].storage_dtype,
                    ))
                }
                _ => None,
            })
            .unwrap_or(entry.representation);
        ops::LogicalTensorMap {
            base: entry.place,
            representation,
            extents: tensor.extents.iter().map(|value| value.raw).collect(),
            steps,
        }
    }
    /// Uniformity of the actual storage address product, including private
    /// participant storage and the operands of its ordered view transforms.
    /// This does not assert that mutable memory is stable across observations.
    pub fn tensor_address_uniformity(&self, tensor: &PortableTensor) -> IntrinsicUniformity {
        let place = self.checked_place(tensor.place);
        let mut values = tensor
            .extents
            .iter()
            .map(|value| value.raw)
            .collect::<Vec<_>>();
        for step in &tensor.steps {
            match step {
                PortableViewStep::Slice(axes) => {
                    for axis in axes {
                        match axis {
                            PortableSliceAxis::Point(value) => values.push(value.raw),
                            PortableSliceAxis::Range { start, end } => {
                                values.extend([start.raw, end.raw])
                            }
                            PortableSliceAxis::Full => {}
                        }
                    }
                }
                PortableViewStep::Reshape { from, to } => {
                    values.extend(from.iter().chain(to).map(|value| value.raw))
                }
                PortableViewStep::Plane { .. } | PortableViewStep::Transpose(_) => {}
            }
        }
        match self.inner.read_uniformity(place.place, values) {
            Uniformity::Workgroup => IntrinsicUniformity::Workgroup,
            Uniformity::Subgroup => IntrinsicUniformity::Subgroup,
            Uniformity::Varying => IntrinsicUniformity::Varying,
        }
    }

    pub fn tensor_plane(&mut self, mut tensor: PortableTensor, plane: u32) -> PortableTensor {
        assert!(!tensor.has_plane(), "plane view was applied twice");
        let entry = self.checked_place(tensor.place);
        let RepresentationKind::Packed(layout) =
            &registry::representation_info(entry.representation).kind
        else {
            panic!("plane projection requires packed storage")
        };
        let schema = &layout.planes[plane as usize];
        let mut axis = entry.rank.checked_sub(1).expect("packed storage rank") as usize;
        for step in &tensor.steps {
            axis = match step {
                PortableViewStep::Slice(axes) => {
                    assert!(
                        !matches!(axes[axis], PortableSliceAxis::Point(_)),
                        "plane projection has no packing axis"
                    );
                    axis - axes[..axis]
                        .iter()
                        .filter(|a| matches!(a, PortableSliceAxis::Point(_)))
                        .count()
                }
                PortableViewStep::Transpose(permutation) => permutation
                    .iter()
                    .position(|a| *a as usize == axis)
                    .expect("packing axis permutation"),
                PortableViewStep::Reshape { .. } => panic!("packed reshape is not a checked view"),
                PortableViewStep::Plane { .. } => unreachable!(),
            };
        }
        // A raw plane contains the owned storage elements of every packet
        // touched by this logical row, including the containing packets of an
        // unaligned slice. The existing ordered view map owns its origin.
        let zero = self.index_constant(0);
        let origins = self.tensor_indices(&tensor, &vec![zero; tensor.extents.len()]);
        let origin = origins[entry.rank as usize - 1];
        let length = tensor.extents[axis];
        let group = self.index_constant(u64::from(layout.group));
        let first_offset = self.binary(BinaryOp::Rem, origin, group);
        let whole = self.binary(BinaryOp::Div, length, group);
        let remainder = self.binary(BinaryOp::Rem, length, group);
        let boundary = self.binary(BinaryOp::Add, first_offset, remainder);
        let boundary_packets = self.index_ceil_div(boundary, group);
        let packets = self.binary(BinaryOp::Add, whole, boundary_packets);
        let one = self.index_constant(1);
        let nonempty = self.binary(BinaryOp::Min, length, one);
        let packets = self.binary(BinaryOp::Mul, packets, nonempty);
        let elements = self.index_constant(u64::from(schema.storage_elements_per_packet()));
        tensor.extents[axis] = self.binary(BinaryOp::Mul, packets, elements);
        tensor.steps.push(PortableViewStep::Plane { plane, axis });
        tensor
    }
    fn index_ceil_div(
        &mut self,
        numerator: PortableValue,
        denominator: PortableValue,
    ) -> PortableValue {
        let quotient = self.binary(BinaryOp::Div, numerator, denominator);
        let remainder = self.binary(BinaryOp::Rem, numerator, denominator);
        let one = self.index_constant(1);
        let extra = self.binary(BinaryOp::Min, remainder, one);
        self.binary(BinaryOp::Add, quotient, extra)
    }
    fn plane_coordinates(
        &mut self,
        entry: PlaceEntry,
        plane: u32,
        axis: usize,
        mut index: Vec<PortableValue>,
    ) -> (Vec<PortableValue>, PortableValue) {
        let RepresentationKind::Packed(layout) =
            &registry::representation_info(entry.representation).kind
        else {
            panic!("plane projection requires packed storage")
        };
        let per_packet = self.index_constant(u64::from(
            layout.planes[plane as usize].storage_elements_per_packet(),
        ));
        let element = self.binary(BinaryOp::Rem, index[axis], per_packet);
        let packet = self.binary(BinaryOp::Div, index[axis], per_packet);
        let group = self.index_constant(u64::from(layout.group));
        index[axis] = self.binary(BinaryOp::Mul, packet, group);
        (index, element)
    }
    fn read_plane_storage(
        &mut self,
        entry: PlaceEntry,
        plane: u32,
        index: &[PortableValue],
        element: PortableValue,
    ) -> PortableValue {
        let RepresentationKind::Packed(layout) =
            &registry::representation_info(entry.representation).kind
        else {
            panic!("plane storage read requires packed storage")
        };
        let ty = dtype_value_type(layout.planes[plane as usize].storage_dtype);
        let indices = self.read_indices(entry, index);
        let element = self.inner.use_value(element.raw, &ValueType::Index);
        let uniformity = self
            .inner
            .read_uniformity(entry.place, indices.iter().copied().chain([element]));
        let out = self.inner.define_with(ty, None, uniformity);
        self.inner.emit(Op::ReadPlane {
            out,
            place: entry.place,
            plane,
            index: indices,
            element,
        });
        PortableValue { raw: out, ty }
    }
    pub fn tensor_slice(
        &mut self,
        tensor: PortableTensor,
        axes: Vec<PortableSliceAxis>,
    ) -> PortableTensor {
        tensor.slice(axes, |end, start| self.binary(BinaryOp::Sub, end, start))
    }
    pub fn tensor_transpose(
        &self,
        tensor: PortableTensor,
        permutation: Vec<u32>,
    ) -> PortableTensor {
        tensor.transpose(permutation)
    }
    pub fn tensor_reshape(
        &self,
        tensor: PortableTensor,
        extents: Vec<PortableValue>,
    ) -> PortableTensor {
        assert!(extents.iter().all(|value| value.ty == ValueType::Index));
        tensor.reshape(extents)
    }
    fn tensor_address(
        &mut self,
        tensor: &PortableTensor,
        index: &[PortableValue],
    ) -> (Vec<PortableValue>, Option<(u32, PortableValue)>) {
        assert_eq!(
            index.len(),
            tensor.extents.len(),
            "logical tensor index rank mismatch"
        );
        let mut index = index.to_vec();
        let mut projection = None;
        for step in tensor.steps.iter().rev() {
            index = match step {
                PortableViewStep::Plane { plane, axis } => {
                    let (mapped, element) = self.plane_coordinates(
                        self.checked_place(tensor.place),
                        *plane,
                        *axis,
                        index,
                    );
                    assert!(projection.replace((*plane, element)).is_none());
                    mapped
                }
                step => {
                    let zero = self.index_constant(0);
                    step.dense_coordinates(&index, zero, |op, a, b| {
                        use crate::tensor_view::CoordinateOp;
                        self.binary(match op {
                            CoordinateOp::Add => BinaryOp::Add,
                            CoordinateOp::Mul => BinaryOp::Mul,
                            CoordinateOp::Div => BinaryOp::Div,
                            CoordinateOp::Rem => BinaryOp::Rem,
                        }, a, b)
                    }).expect("plane handled by representation resolver")
                }
            };
        }
        (index, projection)
    }
    fn tensor_indices(
        &mut self,
        tensor: &PortableTensor,
        index: &[PortableValue],
    ) -> Vec<PortableValue> {
        let (index, projection) = self.tensor_address(tensor, index);
        assert!(projection.is_none(), "plane storage views are read-only");
        index
    }
    pub fn tensor_read(
        &mut self,
        tensor: &PortableTensor,
        index: &[PortableValue],
    ) -> PortableValue {
        let (index, projection) = self.tensor_address(tensor, index);
        match projection {
            None => self.read(tensor.place, &index),
            Some((plane, element)) => {
                self.read_plane_storage(self.checked_place(tensor.place), plane, &index, element)
            }
        }
    }
    pub fn tensor_write(
        &mut self,
        tensor: &PortableTensor,
        index: &[PortableValue],
        value: PortableValue,
    ) {
        assert!(!tensor.has_plane(), "plane views are read-only");
        let index = self.tensor_indices(tensor, index);
        self.write(tensor.place, &index, value);
    }
    pub fn tensor_atomic(
        &mut self,
        op: AtomicOp,
        tensor: &PortableTensor,
        index: &[PortableValue],
        value: PortableValue,
    ) {
        assert!(
            !tensor.has_plane(),
            "plane views are not atomic destinations"
        );
        let index = self.tensor_indices(tensor, index);
        self.atomic(op, tensor.place, &index, value);
    }
    /// Publishes one lane's failed source predicate into the canonical
    /// indexed dense-u32 status word. Atomic max is an order-independent all-lanes OR
    /// and needs no barrier or elected writer.
    pub fn record_source_check(
        &mut self,
        status: &PortableTensor,
        index: PortableValue,
        condition: PortableValue,
    ) {
        assert_eq!(
            condition.ty,
            ValueType::Bool,
            "source check predicate must be boolean"
        );
        assert_eq!(status.extents.len(), 1, "source status is one-dimensional");
        assert!(
            status.steps.is_empty() && !status.has_plane(),
            "source status is an identity dense view"
        );
        let entry = self.checked_place(status.place);
        assert_eq!(
            entry.representation,
            registry::dense(DType::U32),
            "source status uses dense u32 storage"
        );
        let failed = self.not(condition);
        let failed = self.cast(failed, ValueType::Scalar(DType::U32));
        assert_eq!(
            index.ty,
            ValueType::Index,
            "source status index must be an index"
        );
        self.tensor_atomic(AtomicOp::Max, status, &[index], failed);
    }
    pub fn store_slot(&mut self, slot: u32, value: PortableValue) {
        let expected = self
            .inner
            .state
            .result_slots
            .get(slot as usize)
            .expect("portable result slot belongs to the open kernel")
            .kind();
        assert_eq!(
            value.ty,
            expected.value_type(),
            "portable result slot type mismatch"
        );
        assert_eq!(
            self.inner.uniformity_of(value.raw),
            Uniformity::Workgroup,
            "only workgroup-uniform values may cross a kernel/schedule cut"
        );
        let value = self.used(value);
        self.inner.emit(Op::StoreSlot {
            slot,
            value,
            election: ops::StoreElection::GlobalLeader,
        });
    }
    pub fn begin_repeat(
        &mut self,
        start: PortableValue,
        end: PortableValue,
        initial: Vec<PortableValue>,
        recurrence: &[IntrinsicUniformity],
    ) -> (PortableRepeat, PortableValue, Vec<PortableValue>) {
        assert_eq!(start.ty, ValueType::Index);
        assert_eq!(end.ty, ValueType::Index);
        let (start_raw, end_raw) = (self.used(start), self.used(end));
        let schema = ValueSchema::new(initial.iter().map(|value| value.ty.clone()).collect());
        let carries_in = initial
            .into_iter()
            .zip(schema.values())
            .map(|(value, ty)| self.inner.use_value(value.raw, ty))
            .collect::<Vec<_>>();
        let trip = match (self.inner.nat_of(start_raw), self.inner.nat_of(end_raw)) {
            (Some(a), Some(b)) => {
                let m = self.inner.expr.nat_max(b, a);
                Some(self.inner.expr.nat_sub(m, a))
            }
            _ => None,
        };
        let multiplicity = match (
            self.inner.state.blocks[self.inner.block.index() as usize].multiplicity,
            trip,
        ) {
            (Some(a), Some(b)) => Some(self.inner.expr.nat_mul(a, b)),
            _ => None,
        };
        let range_uniformity = self
            .inner
            .uniformity_of(start_raw)
            .combine(self.inner.uniformity_of(end_raw));
        let control_uniformity = self.inner.state.blocks[self.inner.block.index() as usize]
            .control_uniformity
            .combine(range_uniformity);
        let recurrence =
            self.inner
                .recurrence_uniformities(&carries_in, recurrence, control_uniformity);
        let block = self.inner.new_block(multiplicity, control_uniformity);
        let parent = self.inner.block;
        self.inner.block = block;
        let binder = self
            .inner
            .define_with(ValueType::Index, None, range_uniformity);
        let parameters = recurrence
            .iter()
            .zip(schema.values())
            .map(|(uniformity, ty)| self.inner.define_with(ty.clone(), None, *uniformity))
            .collect::<Vec<_>>();
        let carried = parameters
            .iter()
            .copied()
            .zip(schema.values())
            .map(|(raw, ty)| PortableValue {
                raw,
                ty: ty.clone(),
            })
            .collect();
        let token = PortableRepeat {
            parent,
            body: block,
            start: start_raw,
            end: end_raw,
            binder,
            initial: carries_in,
            parameters,
            schema,
            recurrence,
        };
        (
            token,
            PortableValue {
                raw: binder,
                ty: ValueType::Index,
            },
            carried,
        )
    }

    pub fn finish_repeat(
        &mut self,
        token: PortableRepeat,
        next: Vec<PortableValue>,
    ) -> Vec<PortableValue> {
        assert_eq!(
            self.inner.block, token.body,
            "repeat must close its own current body block"
        );
        assert_eq!(
            next.len(),
            token.schema.len(),
            "portable repeat body result count differs from its carry schema"
        );
        let yielded = next
            .iter()
            .zip(token.schema.values())
            .map(|(value, ty)| {
                assert_eq!(&value.ty, ty);
                self.inner.use_value(value.raw, ty)
            })
            .collect::<Vec<_>>();
        self.inner
            .check_recurrence_yields(&yielded, &token.recurrence);
        self.inner.emit(Op::Yield { values: yielded });
        self.inner.block = token.parent;
        let outs = token
            .recurrence
            .iter()
            .zip(token.schema.values())
            .map(|(uniformity, ty)| self.inner.define_with(ty.clone(), None, *uniformity))
            .collect::<Vec<_>>();
        self.inner.emit(Op::Repeat {
            start: token.start,
            end: token.end,
            binder: token.binder,
            carries_in: token.initial,
            carry_params: token.parameters,
            body: token.body,
            outs: outs.clone(),
        });
        outs.into_iter()
            .zip(token.schema.values())
            .map(|(raw, ty)| PortableValue {
                raw,
                ty: ty.clone(),
            })
            .collect()
    }

    pub fn repeat(
        &mut self,
        start: PortableValue,
        end: PortableValue,
        initial: Vec<PortableValue>,
        recurrence: &[IntrinsicUniformity],
        body: impl FnOnce(
            &mut PortableBuilder<'_, B>,
            PortableValue,
            Vec<PortableValue>,
        ) -> Vec<PortableValue>,
    ) -> Vec<PortableValue> {
        let (repeat, binder, carried) = self.begin_repeat(start, end, initial, recurrence);
        let next = body(self, binder, carried);
        self.finish_repeat(repeat, next)
    }
    pub fn begin_branch(&mut self, condition: PortableValue) -> PortableBranch {
        assert_eq!(condition.ty, ValueType::Bool);
        let condition = self.used(condition);
        let parent = self.inner.block;
        let multiplicity = self.inner.state.blocks[parent.index() as usize].multiplicity;
        let control_uniformity = self.inner.state.blocks[parent.index() as usize]
            .control_uniformity
            .combine(self.inner.uniformity_of(condition));
        let then_block = self.inner.new_block(multiplicity, control_uniformity);
        let else_block = self.inner.new_block(multiplicity, control_uniformity);
        self.inner.block = then_block;
        PortableBranch {
            parent,
            then_block,
            else_block,
            condition,
            control_uniformity,
        }
    }

    pub fn begin_otherwise(&mut self, branch: &PortableBranch) {
        assert_eq!(
            self.inner.block, branch.then_block,
            "portable branches must close their nested regions first"
        );
        self.inner.block = branch.else_block;
    }

    pub fn finish_branch(
        &mut self,
        branch: PortableBranch,
        then_values: Vec<PortableValue>,
        else_values: Vec<PortableValue>,
    ) -> Vec<PortableValue> {
        assert_eq!(
            self.inner.block, branch.else_block,
            "portable branch arms must be closed in order"
        );
        let schema = ValueSchema::new(then_values.iter().map(|value| value.ty.clone()).collect());
        assert_eq!(
            else_values
                .iter()
                .map(|value| &value.ty)
                .collect::<Vec<_>>(),
            schema.values().iter().collect::<Vec<_>>(),
            "portable branch result schemas differ"
        );
        self.inner.yield_values(
            branch.then_block,
            &then_values
                .iter()
                .map(|value| value.raw)
                .collect::<Vec<_>>(),
            &schema,
        );
        self.inner.yield_values(
            branch.else_block,
            &else_values
                .iter()
                .map(|value| value.raw)
                .collect::<Vec<_>>(),
            &schema,
        );
        self.inner.block = branch.parent;
        let outs = then_values
            .iter()
            .zip(&else_values)
            .zip(schema.values())
            .map(|((a, b), ty)| {
                let uniformity = branch
                    .control_uniformity
                    .combine(self.inner.uniformity_of(a.raw))
                    .combine(self.inner.uniformity_of(b.raw));
                let left = self.inner.nat_of(a.raw);
                let right = self.inner.nat_of(b.raw);
                let origin = if left == right { left } else { None };
                self.inner.define_with(ty.clone(), origin, uniformity)
            })
            .collect::<Vec<_>>();
        self.inner.emit(Op::Branch {
            cond: branch.condition,
            then: branch.then_block,
            otherwise: branch.else_block,
            outs: outs.clone(),
        });
        outs.into_iter()
            .zip(schema.values())
            .map(|(raw, ty)| PortableValue {
                raw,
                ty: ty.clone(),
            })
            .collect()
    }

    /// Close a branch whose result is a complete product with arm-specific
    /// scalar payload. Missing fields belong only to the other arm; their zero
    /// fillers are transport values, never values of a source expression.
    /// Opaque values must be yielded by both arms with exactly the same type.
    pub fn finish_branch_products(
        &mut self,
        branch: PortableBranch,
        fields: Vec<(Option<PortableValue>, Option<PortableValue>)>,
    ) -> Vec<PortableValue> {
        use ops::ConstantValue;
        assert_eq!(
            self.inner.block, branch.else_block,
            "branch arms must be closed in order"
        );
        let mut then_values = Vec::with_capacity(fields.len());
        let mut else_values = Vec::with_capacity(fields.len());
        for (then, otherwise) in fields {
            let ty = then.or(otherwise).expect("branch field has no producer").ty;
            if let (Some(a), Some(b)) = (then, otherwise) {
                assert_eq!(a.ty, b.ty, "branch product field types differ");
            }
            let filler = |builder: &mut Self, available: PortableValue| {
                if builder
                    .inner
                    .dominates(available.raw.block, builder.inner.block)
                {
                    // An actual pre-branch operand exists in both arms. Reuse
                    // it; an expression-origin annotation alone is not enough
                    // to speculate a fresh, potentially partial calculation.
                    return available;
                }
                let value = match ty {
                    ValueType::Scalar(DType::F32) => ConstantValue::F32(0.0),
                    ValueType::Scalar(DType::F16) => ConstantValue::F16(0),
                    ValueType::Scalar(DType::BF16) => ConstantValue::BF16(0),
                    ValueType::Scalar(DType::I32) => ConstantValue::I32(0),
                    ValueType::Scalar(DType::U32) => ConstantValue::U32(0),
                    ValueType::Scalar(DType::Bool) | ValueType::Bool => ConstantValue::Bool(false),
                    ValueType::Index => ConstantValue::Index(0),
                    ValueType::Opaque { .. } | ValueType::Vector { .. } => {
                        panic!("only scalar branch payload fields may be absent")
                    }
                };
                builder.constant(value, ty)
            };
            self.inner.block = branch.then_block;
            then_values.push(then.unwrap_or_else(|| filler(self, otherwise.unwrap())));
            self.inner.block = branch.else_block;
            else_values.push(otherwise.unwrap_or_else(|| filler(self, then.unwrap())));
        }
        self.finish_branch(branch, then_values, else_values)
    }

    pub fn branch(
        &mut self,
        condition: PortableValue,
        then: impl FnOnce(&mut PortableBuilder<'_, B>) -> Vec<PortableValue>,
        otherwise: impl FnOnce(&mut PortableBuilder<'_, B>) -> Vec<PortableValue>,
    ) -> Vec<PortableValue> {
        let branch = self.begin_branch(condition);
        let then_values = then(self);
        self.begin_otherwise(&branch);
        let else_values = otherwise(self);
        self.finish_branch(branch, then_values, else_values)
    }
    pub fn close(self) -> KernelId {
        assert_eq!(
            self.inner.block.index(),
            0,
            "portable kernel has an unclosed control region"
        );
        self.inner.close()
    }
}

fn a_type<B: PhysicalDialect>(builder: &Builder<'_, B>, value: ErasedValue) -> ValueType {
    builder.entry(value).ty.clone()
}

impl<'a, B: PhysicalDialect> Builder<'a, B> {
    fn allocate_addressable_resource(
        &mut self,
        class_id: crate::target::ResourceClassId,
        units: NatExpr,
        alignment_units: u64,
        lifetime: crate::target::ResourceLifetime,
    ) -> ops::AddressableResourceHandle {
        let class = self
            .resource_classes
            .get(class_id.ordinal() as usize)
            .cloned()
            .expect("resource class id belongs to another target profile");
        assert!(
            alignment_units.is_power_of_two()
                && alignment_units >= class.alignment_units
                && alignment_units % class.alignment_units == 0,
            "resource lease alignment is incompatible with its native class"
        );
        let cursor = self.state.addressable_resource_cursors[class_id.ordinal() as usize];
        let alignment = self.expr.nat(alignment_units);
        let aligned_groups = self.expr.nat_ceil_div(cursor, alignment);
        let offset_units = self.expr.nat_mul(aligned_groups, alignment);
        self.state.addressable_resource_cursors[class_id.ordinal() as usize] =
            self.expr.nat_add(offset_units, units);
        let handle = ops::AddressableResourceHandle {
            owner: self.owner(),
            kernel: self.kernel_index(),
            lease: u32::try_from(self.state.addressable_resources.len())
                .expect("kernel addressable resource lease count exceeds u32"),
        };
        self.state
            .addressable_resources
            .push(ops::AddressableResourceLease {
                handle,
                class_id,
                class,
                offset_units,
                units,
                alignment_units,
                lifetime,
            });
        handle
    }
    fn reborrow(&mut self) -> Builder<'_, B> {
        Builder {
            expr: &mut *self.expr,
            storage: self.storage,
            schedule: self.schedule,
            kernels: &mut *self.kernels,
            state: &mut *self.state,
            target_facts: self.target_facts,
            resource_classes: self.resource_classes,
            vector_support: self.vector_support,
            block: self.block,
        }
    }

    fn in_block(&mut self, block: BlockId) -> Builder<'_, B> {
        let mut sub = self.reborrow();
        sub.block = block;
        sub
    }

    fn kernel_index(&self) -> u32 {
        self.state.kernel
    }

    fn owner(&self) -> OwnerToken {
        self.state.owner
    }

    fn assert_kernel(&self, owner: OwnerToken, kernel: u32) {
        assert_eq!(
            owner,
            self.owner(),
            "kernel handle belongs to another implementation"
        );
        assert_eq!(
            kernel,
            self.kernel_index(),
            "kernel handle belongs to another kernel"
        );
    }

    // ----- values ------------------------------------------------------------

    fn define_with(
        &mut self,
        ty: ValueType,
        nat: Option<NatExpr>,
        uniformity: Uniformity,
    ) -> ErasedValue {
        let index = self.state.values.len() as u32;
        self.state.values.push(ValueEntry {
            ty,
            block: self.block,
            nat,
            uniformity,
        });
        ErasedValue::new(self.owner(), self.kernel_index(), self.block, index)
    }

    fn entry(&self, value: ErasedValue) -> &ValueEntry {
        self.assert_kernel(value.owner, value.kernel);
        match self.state.values.get(value.index() as usize) {
            Some(entry) => entry,
            None => panic!(
                "kernel builder: {value:?} is not a value of the open kernel (a handle of another kernel was used)"
            ),
        }
    }

    fn dominates(&self, dominator: BlockId, block: BlockId) -> bool {
        let mut current = Some(block);
        while let Some(b) = current {
            if b == dominator {
                return true;
            }
            self.assert_kernel(b.owner(), b.kernel());
            current = self.state.blocks[b.index() as usize].parent;
        }
        false
    }

    /// Checks that `value` has type `ty` and is visible in the current block.
    fn use_value(&mut self, value: ErasedValue, ty: &ValueType) -> ErasedValue {
        let entry = self.entry(value);
        if entry.ty != *ty {
            panic!(
                "kernel builder: {value:?} has type {:?} but the operation expects {ty:?} (a handle of another kernel was used)",
                entry.ty
            );
        }
        let defined_in = entry.block;
        if !self.dominates(defined_in, self.block) {
            panic!(
                "kernel builder: {value:?} defined in {defined_in:?} is not visible in {:?}",
                self.block
            );
        }
        value
    }

    fn nat_of(&self, value: ErasedValue) -> Option<NatExpr> {
        self.entry(value).nat
    }

    fn uniformity_of(&self, value: ErasedValue) -> Uniformity {
        self.entry(value).uniformity
    }

    fn read_uniformity(
        &self,
        place: PlaceRef,
        indices: impl IntoIterator<Item = ErasedValue>,
    ) -> Uniformity {
        let storage = match place {
            PlaceRef::Global { .. } => Uniformity::Workgroup,
            PlaceRef::Local { index } => match self.state.locals[index as usize].kind {
                LaunchLocalKind::Workgroup => Uniformity::Workgroup,
                LaunchLocalKind::Participant | LaunchLocalKind::Register => Uniformity::Varying,
            },
        };
        storage
            .combine(self.combined_uniformity(indices))
            .combine(self.state.blocks[self.block.index() as usize].control_uniformity)
    }
    fn recurrence_uniformities(
        &self,
        initial: &[ErasedValue],
        declared: &[IntrinsicUniformity],
        control: Uniformity,
    ) -> Vec<Uniformity> {
        assert_eq!(
            initial.len(),
            declared.len(),
            "loop recurrence uniformity count differs from carry schema"
        );
        initial
            .iter()
            .zip(declared)
            .map(|(initial, declared)| {
                let uniformity = Uniformity::from_intrinsic(*declared).combine(control);
                assert!(
                    self.uniformity_of(*initial) <= uniformity,
                    "loop initial value exceeds its recurrence uniformity"
                );
                uniformity
            })
            .collect()
    }
    fn check_recurrence_yields(&self, values: &[ErasedValue], recurrence: &[Uniformity]) {
        assert_eq!(
            values.len(),
            recurrence.len(),
            "loop yield count differs from recurrence schema"
        );
        for (value, expected) in values.iter().zip(recurrence) {
            assert!(
                self.uniformity_of(*value) <= *expected,
                "loop yielded value exceeds its recurrence uniformity"
            );
        }
    }

    fn combined_uniformity(&self, values: impl IntoIterator<Item = ErasedValue>) -> Uniformity {
        values
            .into_iter()
            .fold(Uniformity::Workgroup, |result, value| {
                result.combine(self.uniformity_of(value))
            })
    }

    fn emit(&mut self, op: Op<B>) {
        self.state.blocks[self.block.index() as usize].ops.push(op);
    }

    // ----- places ------------------------------------------------------------

    fn push_place(&mut self, place: PlaceRef, representation: RepresentationId, rank: u32) -> u32 {
        let index = self.state.places.len() as u32;
        self.state.places.push(PlaceEntry {
            place,
            representation,
            rank,
        });
        index
    }

    fn bind_view(
        &mut self,
        view: crate::storage::AnyBufferView,
        access: BindingAccess,
    ) -> (BindingSlot, u32) {
        assert_eq!(
            view.owner(),
            self.owner(),
            "buffer view belongs to another implementation"
        );
        let extents = match self.storage.views().get(view.index as usize) {
            Some(layout) => layout.extents.clone(),
            None => panic!(
                "kernel builder: {view:?} is not a view of the implementation under construction"
            ),
        };
        let rank = extents.len() as u32;
        let slot = BindingSlot::new(
            self.owner(),
            self.kernel_index(),
            self.state.bindings.len() as u32,
        );
        self.state.bindings.push(Binding {
            slot,
            view,
            access,
            rank,
            extents,
        });
        (slot, rank)
    }

    fn geometry(&mut self, kind: GeometryValue) -> ErasedValue {
        let uniformity = match kind {
            GeometryValue::WorkgroupId(_)
            | GeometryValue::WorkgroupSize(_)
            | GeometryValue::GridSize(_)
            | GeometryValue::SubgroupSize => Uniformity::Workgroup,
            GeometryValue::SubgroupOrdinal => Uniformity::Subgroup,
            GeometryValue::LocalId(_)
            | GeometryValue::GlobalId(_)
            | GeometryValue::SubgroupLane => Uniformity::Varying,
        };
        let out = self.define_with(ValueType::Index, None, uniformity);
        self.emit(Op::Geometry { out, kind });
        out
    }
    /// The physical binary constructor.
    fn binary_value(
        &mut self,
        op: BinaryOp,
        ty: ValueType,
        a: ErasedValue,
        b: ErasedValue,
    ) -> ErasedValue {
        let nat = match (self.nat_of(a), self.nat_of(b)) {
            (Some(x), Some(y)) if ty == ValueType::Index => Some(match op {
                BinaryOp::Add | BinaryOp::Sub | BinaryOp::Mul => {
                    // Index is the selected native u64 word, not an unbounded
                    // source quantity. Derive its exact modular result with
                    // mathematical intermediates before projecting back to u64.
                    let maximum = self.expr.nat(u64::MAX);
                    let one = self.expr.nat(1);
                    let modulus = self.expr.nat_add(maximum, one);
                    let result = match op {
                        BinaryOp::Add => self.expr.nat_add(x, y),
                        BinaryOp::Sub => {
                            let shifted = self.expr.nat_add(x, modulus);
                            self.expr.nat_sub(shifted, y)
                        }
                        BinaryOp::Mul => self.expr.nat_mul(x, y),
                        _ => unreachable!(),
                    };
                    self.expr.nat_rem(result, modulus)
                }
                BinaryOp::Div => self.expr.nat_div(x, y),
                BinaryOp::Rem => self.expr.nat_rem(x, y),
                BinaryOp::Min => self.expr.nat_min(x, y),
                BinaryOp::Max => self.expr.nat_max(x, y),
            }),
            _ => None,
        };
        let uniformity = self.uniformity_of(a).combine(self.uniformity_of(b));
        let out = self.define_with(ty, nat, uniformity);
        self.emit(Op::Binary { op, out, a, b });
        out
    }
    pub(super) fn barrier(&mut self, s: BarrierScope) {
        let control = self.state.blocks[self.block.index() as usize].control_uniformity;
        let legal = match s {
            BarrierScope::Workgroup => control == Uniformity::Workgroup,
            BarrierScope::Subgroup => control <= Uniformity::Subgroup,
        };
        assert!(
            legal,
            "kernel builder: {s:?} barrier is nested under divergent lexical control"
        );
        self.state.barriers += 1;
        if s == BarrierScope::Subgroup {
            self.state.uses_subgroup = true;
        }
        self.emit(Op::Barrier(s));
    }
    fn new_block(
        &mut self,
        multiplicity: Option<NatExpr>,
        control_uniformity: Uniformity,
    ) -> BlockId {
        let id = BlockId::new(
            self.owner(),
            self.kernel_index(),
            self.state.blocks.len() as u32,
        );
        self.state.blocks.push(BlockData {
            ops: Vec::new(),
            parent: Some(self.block),
            multiplicity,
            control_uniformity,
        });
        id
    }

    fn yield_values(&mut self, block: BlockId, values: &[ErasedValue], schema: &ValueSchema) {
        assert_eq!(
            values.len(),
            schema.len(),
            "branch/repeat result count differs from its typed schema"
        );
        let mut sub = self.in_block(block);
        let values: Vec<ErasedValue> = values
            .iter()
            .zip(schema.values().iter())
            .map(|(v, ty)| sub.use_value(*v, ty))
            .collect();
        sub.emit(Op::Yield { values });
    }

    pub(super) fn close(self) -> KernelId {
        let owner = self.owner();
        let kernel_index = self.kernel_index();
        let root = BlockId::new(owner, kernel_index, 0);
        let written = definitely_written(&self.state.blocks, root);
        for (slot, _) in self.state.result_slots.iter().enumerate() {
            if !written.contains(&(slot as u32)) {
                panic!("kernel builder: result slot #{slot} is not written on every path");
            }
        }
        let uses_subgroup = self.state.uses_subgroup;
        let interface = KernelInterface {
            bindings: std::mem::take(&mut self.state.bindings),
            nat_args: std::mem::take(&mut self.state.nat_args),
            scalar_args: std::mem::take(&mut self.state.scalar_args),
            result_slots: std::mem::take(&mut self.state.result_slots),
            uses_subgroup,
        };
        let resource_facts = ResourceFacts {
            local_kinds: self.state.locals.iter().map(|l| l.kind).collect(),
            barriers: self.state.barriers,
            uses_subgroup,
            binding_count: interface.bindings.len() as u32,
        };
        let mut resource_references = vec![0u32; self.state.addressable_resources.len()];
        for block in &self.state.blocks {
            for op in &block.ops {
                if let Op::Intrinsic { op, .. } = op {
                    for handle in B::intrinsic_addressable_resources(op) {
                        self.assert_kernel(handle.owner, handle.kernel);
                        let count = resource_references
                            .get_mut(handle.lease as usize)
                            .expect("intrinsic references an absent addressable-resource lease");
                        *count = count
                            .checked_add(1)
                            .expect("addressable-resource reference count overflow");
                    }
                }
            }
        }
        for (lease, references) in self
            .state
            .addressable_resources
            .iter()
            .zip(resource_references)
        {
            assert!(
                references != 0,
                "addressable-resource lease has no intrinsic owner"
            );
            if lease.lifetime == crate::target::ResourceLifetime::Operation {
                assert_eq!(
                    references, 1,
                    "operation-lifetime addressable resource is referenced by more than one intrinsic"
                );
            }
        }
        let blocks = std::mem::take(&mut self.state.blocks)
            .into_iter()
            .map(|b| (Block { ops: b.ops }, b.multiplicity))
            .collect::<Vec<_>>();
        let (blocks, block_multiplicity): (Vec<_>, Vec<_>) = blocks.into_iter().unzip();
        let id = KernelId::new(owner, kernel_index);
        self.kernels.push(Kernel {
            inner: KernelData {
                owner,
                kernel: kernel_index,
                interface,
                locals: std::mem::take(&mut self.state.locals),
                blocks,
                block_multiplicity,
                values: std::mem::take(&mut self.state.values),
                intrinsic_resources: std::mem::take(&mut self.state.intrinsic_resources),
                addressable_resources: std::mem::take(&mut self.state.addressable_resources),
                intrinsics_used: std::mem::take(&mut self.state.intrinsics_used),
                resource_facts,
            },
        });
        id
    }
}

/// Result slots written on every path through `block` (structural: a
/// branch writes what both arms write; a repeat writes nothing because it
/// may run zero times).
fn definitely_written<B: PhysicalDialect>(blocks: &[BlockData<B>], block: BlockId) -> Vec<u32> {
    let mut written = Vec::new();
    for op in &blocks[block.index() as usize].ops {
        match op {
            Op::StoreSlot { slot, .. } => written.push(*slot),
            Op::Branch {
                then, otherwise, ..
            } => {
                let a = definitely_written(blocks, *then);
                let b = definitely_written(blocks, *otherwise);
                written.extend(a.into_iter().filter(|s| b.contains(s)));
            }
            _ => {}
        }
    }
    written
}

fn representation_alignment(id: RepresentationId) -> u64 {
    match &registry::representation_info(id).kind {
        RepresentationKind::Dense(dtype) => dtype.bytes() as u64,
        RepresentationKind::Packed(layout) => u64::from(layout.packet_alignment),
        RepresentationKind::External(layout) => u64::from(layout.packet_alignment),
    }
}

// Registry-validated dispatch used only by the core semantic walker for an
// authored backend lowering/helper. Result shape comes from the registry;
// the backend cannot invent it.
impl<'s, 'k, B: PhysicalDialect> ops::SemanticIntrinsicSink<'s, 'k, B> {
    /// Allocates compiler-owned workgroup scratch for an authored intrinsic.
    /// The returned logical map is the only addressable path: storage size,
    /// alignment, lifetime and native binding remain in the ordinary local
    /// topology rather than a backend-private byte reservation.
    pub fn workgroup_tensor(
        &mut self,
        representation: RepresentationId,
        extents: Vec<NatExpr>,
    ) -> ops::LogicalTensorMap {
        let tensor = self
            .builder
            .local_tensor(LaunchLocalKind::Workgroup, representation, extents);
        self.builder.tensor_mapping(&tensor)
    }

    pub fn addressable_resource(
        &mut self,
        class: crate::target::ResourceClassId,
        units: NatExpr,
        alignment_units: u64,
        lifetime: crate::target::ResourceLifetime,
    ) -> ops::AddressableResourceHandle {
        self.builder
            .inner
            .allocate_addressable_resource(class, units, alignment_units, lifetime)
    }
    pub fn open(
        builder: &'s mut PortableBuilder<'k, B>,
        call: &ops::SemanticIntrinsicCall<'_>,
    ) -> Self {
        let owned = matches!(call.signature.result, IntrinsicResultType::Owned { .. });
        assert_eq!(
            call.destination.is_some(),
            owned,
            "intrinsic destination differs from registered result"
        );
        let mut semantic_arguments = Vec::new();
        let mut mapping_dependencies = Vec::new();
        for operand in call.operands {
            match operand {
                ops::SemanticIntrinsicOperand::Scalar(value)
                | ops::SemanticIntrinsicOperand::Constant(value) => {
                    semantic_arguments.push(builder.used(value.value));
                }
                ops::SemanticIntrinsicOperand::Opaque(value) => {
                    semantic_arguments.push(builder.used(value.value));
                }
                ops::SemanticIntrinsicOperand::Readable(place)
                | ops::SemanticIntrinsicOperand::Writable(place) => {
                    let mapping = builder.tensor_mapping(&place.tensor);
                    for dependency in mapping.dependencies() {
                        builder.inner.use_value(dependency, &ValueType::Index);
                        if !mapping_dependencies.contains(&dependency) {
                            mapping_dependencies.push(dependency);
                        }
                    }
                }
            }
        }
        if let Some(destination) = &call.destination {
            let mapping = builder.tensor_mapping(&destination.tensor);
            for dependency in mapping.dependencies() {
                builder.inner.use_value(dependency, &ValueType::Index);
                if !mapping_dependencies.contains(&dependency) {
                    mapping_dependencies.push(dependency);
                }
            }
        }
        Self {
            builder,
            signature: call.signature.clone(),
            emitted: false,
            result: None,
            semantic_arguments,
            mapping_dependencies,
        }
    }

    pub fn scalar(&mut self, operand: ops::SemanticIntrinsicOperand) -> ErasedValue {
        let value = match operand {
            ops::SemanticIntrinsicOperand::Scalar(value)
            | ops::SemanticIntrinsicOperand::Constant(value) => value.value,
            _ => panic!("intrinsic dispatcher projected a non-scalar operand as scalar"),
        };
        self.builder.used(value)
    }

    pub fn readable(&mut self, operand: ops::SemanticIntrinsicOperand) -> ops::LogicalTensorMap {
        let tensor = match operand {
            ops::SemanticIntrinsicOperand::Readable(place)
            | ops::SemanticIntrinsicOperand::Writable(place) => place.tensor,
            _ => panic!("intrinsic dispatcher projected a non-place operand as readable"),
        };
        let mapping = self.builder.tensor_mapping(&tensor);
        mapping
    }

    pub fn writable(&mut self, operand: ops::SemanticIntrinsicOperand) -> ops::LogicalTensorMap {
        let tensor = match operand {
            ops::SemanticIntrinsicOperand::Writable(place)
                if place.tensor.place.write == PortableWriteCapability::DenseElement =>
            {
                place.tensor
            }
            _ => panic!("intrinsic dispatcher projected a non-writable operand as writable"),
        };
        let mapping = self.builder.tensor_mapping(&tensor);
        mapping
    }

    pub fn opaque(&mut self, operand: ops::SemanticIntrinsicOperand) -> ErasedValue {
        let value = match operand {
            ops::SemanticIntrinsicOperand::Opaque(value) => value.value,
            _ => panic!("intrinsic dispatcher projected a non-opaque operand as opaque"),
        };
        self.builder.used(value)
    }

    pub fn nat(&mut self, value: u64) -> NatExpr {
        self.builder.inner.expr.nat(value)
    }

    pub fn nat_mul(&mut self, a: NatExpr, b: NatExpr) -> NatExpr {
        self.builder.inner.expr.nat_mul(a, b)
    }

    pub fn nat_ceil_div(&mut self, a: NatExpr, b: NatExpr) -> NatExpr {
        self.builder.inner.expr.nat_ceil_div(a, b)
    }

    pub fn index(&mut self, value: NatExpr) -> ErasedValue {
        self.builder.nat_arg(value).raw
    }

    pub fn index_value(&mut self, value: ErasedValue) -> ErasedValue {
        self.builder.inner.use_value(value, &ValueType::Index)
    }

    pub fn bool(&mut self, value: bool) -> seismic_lang::expr::BoolExpr {
        self.builder.inner.expr.bool(value)
    }

    /// Canonical logical extents of a place operand. Geometry decisions are
    /// expressed from these arena values and therefore enter the same hard
    /// constraints and executable evaluators as every other launch fact.
    pub fn extents(&self, operand: ops::SemanticIntrinsicOperand) -> Vec<ErasedValue> {
        let tensor = match operand {
            ops::SemanticIntrinsicOperand::Readable(place)
            | ops::SemanticIntrinsicOperand::Writable(place) => place.tensor,
            _ => panic!("intrinsic dispatcher requested extents of a non-place operand"),
        };
        tensor.extents.into_iter().map(|value| value.raw).collect()
    }

    pub fn emit(
        &mut self,
        op: B::Intrinsic,
        resources: IntrinsicResources,
        destination: Option<ops::SemanticPlace>,
    ) {
        assert!(!self.emitted, "intrinsic dispatcher emitted twice");
        self.emitted = true;
        if resources.requires_subgroup {
            self.builder.inner.state.uses_subgroup = true;
        }
        self.builder.inner.state.intrinsic_resources.push(resources);
        self.builder
            .inner
            .state
            .intrinsics_used
            .push(self.signature.id);
        let uniformity = match self.signature.effects.result_uniformity {
            IntrinsicUniformity::Workgroup => Uniformity::Workgroup,
            IntrinsicUniformity::Subgroup => Uniformity::Subgroup,
            IntrinsicUniformity::Varying => Uniformity::Varying,
        };
        let result = match &self.signature.result {
            IntrinsicResultType::Void => {
                assert!(destination.is_none());
                self.builder.inner.emit(Op::Intrinsic {
                    intrinsic: self.signature.id,
                    op,
                    outs: Vec::new(),
                    args: self.semantic_arguments.clone(),
                    mapping_dependencies: self.mapping_dependencies.clone(),
                });
                ops::SemanticIntrinsicResult::Void
            }
            IntrinsicResultType::Scalar(dtype) => {
                assert!(destination.is_none());
                let ty = dtype_value_type(*dtype);
                let out = self.builder.inner.define_with(ty.clone(), None, uniformity);
                self.builder.inner.emit(Op::Intrinsic {
                    intrinsic: self.signature.id,
                    op,
                    outs: vec![out],
                    args: self.semantic_arguments.clone(),
                    mapping_dependencies: self.mapping_dependencies.clone(),
                });
                ops::SemanticIntrinsicResult::Scalar(ops::SemanticScalar {
                    value: PortableValue { raw: out, ty },
                    dtype: *dtype,
                    index: false,
                    uniformity: self.signature.effects.result_uniformity,
                })
            }
            IntrinsicResultType::Owned { .. } => {
                let destination = destination.expect("owned intrinsic result lacks destination");
                self.builder.inner.emit(Op::Intrinsic {
                    intrinsic: self.signature.id,
                    op,
                    outs: Vec::new(),
                    args: self.semantic_arguments.clone(),
                    mapping_dependencies: self.mapping_dependencies.clone(),
                });
                ops::SemanticIntrinsicResult::Owned(destination)
            }
            IntrinsicResultType::Opaque { capability, name } => {
                assert!(destination.is_none());
                let ty = ValueType::Opaque { name };
                let out = self.builder.inner.define_with(ty.clone(), None, uniformity);
                self.builder.inner.emit(Op::Intrinsic {
                    intrinsic: self.signature.id,
                    op,
                    outs: vec![out],
                    args: self.semantic_arguments.clone(),
                    mapping_dependencies: self.mapping_dependencies.clone(),
                });
                ops::SemanticIntrinsicResult::Opaque(ops::SemanticOpaque {
                    value: PortableValue { raw: out, ty },
                    capability: *capability,
                    name,
                    uniformity: self.signature.effects.result_uniformity,
                })
            }
        };
        self.result = Some(result);
    }

    pub fn finish(self) -> ops::SemanticIntrinsicResult {
        assert!(
            self.emitted,
            "intrinsic dispatcher returned without emitting"
        );
        self.result
            .expect("emitted intrinsic has one closed result")
    }
}

// ---------------------------------------------------------------------------
// Closed kernels
// ---------------------------------------------------------------------------

#[derive(Debug)]
pub(crate) struct Arena<B: PhysicalDialect> {
    owner: OwnerToken,
    kernels: Vec<Kernel<B>>,
}

impl<B: PhysicalDialect> Arena<B> {
    pub(super) fn get(&self, id: KernelId) -> Option<&Kernel<B>> {
        (id.owner() == self.owner)
            .then(|| self.kernels.get(id.index() as usize))
            .flatten()
    }
    pub(super) fn retained_bytes(&self) -> usize {
        self.kernels.capacity() * std::mem::size_of::<Kernel<B>>()
            + self
                .kernels
                .iter()
                .map(|kernel| kernel.inner.retained_bytes())
                .sum::<usize>()
    }
    pub(super) fn kernels(&self) -> impl Iterator<Item = (KernelId, &Kernel<B>)> + '_ {
        self.kernels
            .iter()
            .enumerate()
            .map(|(i, k)| (KernelId::new(self.owner, i as u32), k))
    }
    pub(super) fn kernel(&self, id: KernelId) -> &Kernel<B> {
        assert_eq!(
            id.owner(),
            self.owner,
            "kernel handle belongs to another implementation"
        );
        match self.kernels.get(id.index() as usize) {
            Some(kernel) => kernel,
            None => panic!(
                "{id:?} is outside its kernel arena of {} kernels",
                self.kernels.len()
            ),
        }
    }
}

pub(crate) fn arena_from_kernels<B: PhysicalDialect>(
    owner: OwnerToken,
    kernels: Vec<Kernel<B>>,
) -> KernelArena<B> {
    KernelArena {
        inner: Arena { owner, kernels },
    }
}

pub(crate) fn arena_into_kernels<B: PhysicalDialect>(arena: KernelArena<B>) -> Vec<Kernel<B>> {
    arena.inner.kernels
}

pub(crate) fn data<B: PhysicalDialect>(kernel: &Kernel<B>) -> &KernelData<B> {
    &kernel.inner
}

#[derive(Debug)]
pub(crate) struct KernelData<B: PhysicalDialect> {
    pub(super) owner: OwnerToken,
    pub(super) kernel: u32,
    interface: KernelInterface,
    locals: Vec<LocalAllocation>,
    blocks: Vec<Block<B>>,
    block_multiplicity: Vec<Option<NatExpr>>,
    values: Vec<ValueEntry>,
    intrinsic_resources: Vec<IntrinsicResources>,
    addressable_resources: Vec<ops::AddressableResourceLease>,
    intrinsics_used: Vec<IntrinsicId>,
    resource_facts: ResourceFacts,
}

impl<B: PhysicalDialect> KernelData<B> {
    fn retained_bytes(&self) -> usize {
        let interface = &self.interface;
        interface.bindings.capacity() * std::mem::size_of::<ops::Binding>()
            + interface
                .bindings
                .iter()
                .map(|binding| binding.extents.capacity() * std::mem::size_of::<NatExpr>())
                .sum::<usize>()
            + interface.nat_args.capacity() * std::mem::size_of::<NatExpr>()
            + interface.scalar_args.capacity()
                * std::mem::size_of::<(SymbolId, crate::repr::ScalarKind)>()
            + interface.result_slots.capacity()
                * std::mem::size_of::<crate::schedule::AnyScalarSlot>()
            + self.locals.capacity() * std::mem::size_of::<LocalAllocation>()
            + self
                .locals
                .iter()
                .map(|local| local.extents.capacity() * std::mem::size_of::<NatExpr>())
                .sum::<usize>()
            + self.blocks.capacity() * std::mem::size_of::<Block<B>>()
            + self
                .blocks
                .iter()
                .map(|block| block.ops.capacity() * std::mem::size_of::<Op<B>>())
                .sum::<usize>()
            + self.block_multiplicity.capacity() * std::mem::size_of::<Option<NatExpr>>()
            + self.values.capacity() * std::mem::size_of::<ValueEntry>()
            + self.intrinsic_resources.capacity() * std::mem::size_of::<IntrinsicResources>()
            + self.addressable_resources.capacity()
                * std::mem::size_of::<ops::AddressableResourceLease>()
            + self.intrinsics_used.capacity() * std::mem::size_of::<IntrinsicId>()
    }
    pub(crate) fn interface(&self) -> &KernelInterface {
        &self.interface
    }
    pub(crate) fn locals(&self) -> &[LocalAllocation] {
        &self.locals
    }
    pub(super) fn root(&self) -> BlockId {
        BlockId::new(self.owner, self.kernel, 0)
    }
    pub(super) fn block(&self, id: BlockId) -> &Block<B> {
        assert_eq!(
            id.owner(),
            self.owner,
            "block belongs to another implementation"
        );
        assert_eq!(id.kernel(), self.kernel, "block belongs to another kernel");
        match self.blocks.get(id.index() as usize) {
            Some(block) => block,
            None => panic!(
                "{id:?} is outside its kernel of {} blocks",
                self.blocks.len()
            ),
        }
    }
    fn value(&self, v: ErasedValue) -> &ValueEntry {
        assert_eq!(
            v.owner, self.owner,
            "value belongs to another implementation"
        );
        assert_eq!(v.kernel, self.kernel, "value belongs to another kernel");
        match self.values.get(v.index() as usize) {
            Some(value) => {
                assert_eq!(v.block, value.block, "value carries the wrong defining block");
                value
            },
            None => panic!(
                "{v:?} is outside its kernel of {} values",
                self.values.len()
            ),
        }
    }
    pub(super) fn value_type(&self, value: ErasedValue) -> ValueType {
        self.value(value).ty.clone()
    }
    pub(super) fn exact_nat(&self, value: ErasedValue) -> Option<NatExpr> {
        self.value(value).nat
    }
    pub(crate) fn resource_facts(&self) -> &ResourceFacts {
        &self.resource_facts
    }

    // ----- crate-private facts for the implementation builder ----------------

    pub(crate) fn intrinsic_resources(&self) -> &[IntrinsicResources] {
        &self.intrinsic_resources
    }
    pub(crate) fn addressable_resources(&self) -> &[ops::AddressableResourceLease] {
        &self.addressable_resources
    }
    pub(crate) fn intrinsics_used(&self) -> &[IntrinsicId] {
        &self.intrinsics_used
    }
    pub(crate) fn blocks(&self) -> &[Block<B>] {
        &self.blocks
    }
    pub(crate) fn block_multiplicity(&self) -> &[Option<NatExpr>] {
        &self.block_multiplicity
    }
    pub(crate) fn value_types(&self) -> impl ExactSizeIterator<Item = &ValueType> {
        self.values.iter().map(|value| &value.ty)
    }
    /// Imports a closed child kernel into another implementation. All
    /// owner-qualified references are rewritten together in this one local
    /// operation; no durable old-to-new side table survives it.
    pub(crate) fn rebrand(
        &mut self,
        owner: OwnerToken,
        kernel: u32,
        map_view: impl Fn(crate::storage::AnyBufferView) -> crate::storage::AnyBufferView + Copy,
        map_slot: impl Fn(AnyScalarSlot) -> AnyScalarSlot + Copy,
    ) {
        let old_owner = self.owner;
        let old_kernel = self.kernel;
        let block = |id: BlockId| {
            assert_eq!(id.owner(), old_owner);
            assert_eq!(id.kernel(), old_kernel);
            BlockId::new(owner, kernel, id.index())
        };
        let value = |v: ErasedValue| {
            assert_eq!(v.owner, old_owner);
            assert_eq!(v.kernel, old_kernel);
            ErasedValue::new(owner, kernel, block(v.block), v.index())
        };
        let slot = |s: BindingSlot| {
            assert_eq!(s.owner(), old_owner);
            assert_eq!(s.kernel(), old_kernel);
            BindingSlot::new(owner, kernel, s.index())
        };
        let place = |p: PlaceRef| match p {
            PlaceRef::Global { slot: old } => PlaceRef::Global { slot: slot(old) },
            PlaceRef::Local { index } => PlaceRef::Local { index },
        };
        for binding in &mut self.interface.bindings {
            binding.slot = slot(binding.slot);
            binding.view = map_view(binding.view);
        }
        for result in &mut self.interface.result_slots {
            *result = map_slot(*result);
        }
        for entry in &mut self.values {
            entry.block = block(entry.block);
        }
        for block_data in &mut self.blocks {
            for op in &mut block_data.ops {
                rebrand_op(op, value, block, place);
            }
        }
        self.owner = owner;
        self.kernel = kernel;
    }
}

fn rebrand_op<B: PhysicalDialect>(
    op: &mut Op<B>,
    value: impl Fn(ErasedValue) -> ErasedValue + Copy,
    block: impl Fn(BlockId) -> BlockId + Copy,
    place: impl Fn(PlaceRef) -> PlaceRef + Copy,
) {
    let values = |items: &mut Vec<ErasedValue>| {
        for item in items {
            *item = value(*item);
        }
    };
    match op {
        Op::Constant { out, .. }
        | Op::Geometry { out, .. }
        | Op::NatArg { out, .. }
        | Op::ScalarArg { out, .. } => *out = value(*out),
        Op::Binary { out, a, b, .. }
        | Op::Bit { out, a, b, .. }
        | Op::VectorBinary { out, a, b, .. }
        | Op::VectorBit { out, a, b, .. }
        | Op::Cmp { out, a, b, .. }
        | Op::Logic { out, a, b, .. } => {
            *out = value(*out);
            *a = value(*a);
            *b = value(*b);
        }
        Op::Unary { out, a, .. }
        | Op::Math { out, a, .. }
        | Op::Cast { out, a, .. }
        | Op::Bitcast { out, a, .. }
        | Op::ScalarBits { out, a }
        | Op::ScalarFromBits { out, a }
        | Op::VectorSplat { out, value: a }
        | Op::VectorUnary { out, a, .. }
        | Op::VectorCast { out, a, .. }
        | Op::VectorLane { out, vector: a, .. }
        | Op::VectorReduceAdd { out, vector: a }
        | Op::Not { out, a } => {
            *out = value(*out);
            *a = value(*a);
        }
        Op::Fma { out, a, b, c } | Op::VectorFma { out, a, b, c } => {
            *out = value(*out);
            *a = value(*a);
            *b = value(*b);
            *c = value(*c);
        }
        Op::Select { out, cond, a, b } => {
            *out = value(*out);
            *cond = value(*cond);
            *a = value(*a);
            *b = value(*b);
        }
        Op::Read {
            out,
            place: p,
            index,
            ..
        }
        | Op::ReadPlaneField {
            out,
            place: p,
            index,
            ..
        } => {
            *out = value(*out);
            *p = place(*p);
            values(index);
        }
        Op::ReadPlane {
            out,
            place: p,
            index,
            element,
            ..
        } => {
            *out = value(*out);
            *p = place(*p);
            values(index);
            *element = value(*element);
        }
        Op::VectorFromLanes { out, lanes } => {
            *out = value(*out);
            values(lanes);
        }
        Op::VectorRead {
            out,
            place: p,
            index,
            active,
            ..
        } => {
            *out = value(*out);
            *p = place(*p);
            values(index);
            *active = value(*active);
        }
        Op::VectorWrite {
            place: p,
            index,
            active,
            value: stored,
            ..
        } => {
            *p = place(*p);
            values(index);
            *active = value(*active);
            *stored = value(*stored);
        }
        Op::RepresentationConvertPacket {
            source,
            destination,
            packet,
            ..
        } => {
            *source = match place(PlaceRef::Global { slot: *source }) {
                PlaceRef::Global { slot } => slot,
                PlaceRef::Local { .. } => {
                    panic!("global conversion source remapped to a local place")
                }
            };
            *destination = match place(PlaceRef::Global { slot: *destination }) {
                PlaceRef::Global { slot } => slot,
                PlaceRef::Local { .. } => {
                    panic!("global conversion destination remapped to a local place")
                }
            };
            *packet = value(*packet);
        }
        Op::Write {
            place: p,
            index,
            value: input,
            ..
        }
        | Op::Atomic {
            place: p,
            index,
            value: input,
            ..
        } => {
            *p = place(*p);
            values(index);
            *input = value(*input);
        }
        Op::Extent { out, place: p, .. } => {
            *out = value(*out);
            *p = place(*p);
        }
        Op::StoreSlot { value: input, .. } => *input = value(*input),
        Op::Barrier(_) => {}
        Op::Intrinsic {
            outs,
            args,
            mapping_dependencies,
            ..
        } => {
            values(outs);
            values(args);
            values(mapping_dependencies);
        }
        Op::Branch {
            cond,
            then,
            otherwise,
            outs,
        } => {
            *cond = value(*cond);
            *then = block(*then);
            *otherwise = block(*otherwise);
            values(outs);
        }
        Op::Repeat {
            start,
            end,
            binder,
            carries_in,
            carry_params,
            body,
            outs,
        } => {
            *start = value(*start);
            *end = value(*end);
            *binder = value(*binder);
            values(carries_in);
            values(carry_params);
            *body = block(*body);
            values(outs);
        }
        Op::Yield { values: yielded } => values(yielded),
    }
}

/// Mutable access to a kernel's data for splicing.
pub(crate) fn data_mut<B: PhysicalDialect>(kernel: &mut Kernel<B>) -> &mut KernelData<B> {
    &mut kernel.inner
}

impl PortableValue {
    pub fn raw(self) -> ErasedValue {
        self.raw
    }
    pub fn ty(self) -> ValueType {
        self.ty
    }
}

pub(crate) fn arena_owner<B: PhysicalDialect>(arena: &KernelArena<B>) -> OwnerToken {
    arena.inner.owner
}
