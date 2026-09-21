//! W4-owned kernel builder and arena internals. The public builder in
//! `kernel/mod.rs` is frozen; these signatures serve every public method.
//!
//! Construction facts established here and never re-checked downstream:
//! - every `ErasedValue` of a closed kernel is dense and typed;
//! - every operand of an op is defined in the op's block or a dominating
//!   block (lexical scoping, §7.2);
//! - every join/carry has one typed schema (by the `KernelValues` type of
//!   the branch/repeat call);
//! - every read/write index has the rank of its place;
//! - every declared result slot is written on every path;
//! - every intrinsic's resources and numerical contract are recorded.
//!
//! The only panics are factory authoring bugs against the private builder
//! (a handle of another kernel, a value used outside its scope, a rank
//! mismatch, an unwritten result slot, an unsupported plane) and are
//! §13.3.2 (private arena id) / §13.3.1 (registry) categories.

use super::ops::{
    self, BarrierScope, BinaryOp, Binding, BindingAccess, BitOp, Block, CmpOp, ErasedValue,
    GeometryValue, IntrinsicResources, IntrinsicSink, KernelInterface, LogicOp, MathPrecision,
    NumericalFact, Op, PlaceRef, ResourceFacts, UnaryOp, ValueSchema, ValueType,
};
use super::{
    BindingSlot, BlockId, Kernel, KernelArena, KernelBuilder, KernelId, KernelValues, PlaneId,
    ReadablePlaceId, ScalarId, TypedIntrinsic, VectorId, WritablePlaceId, WritableScalar,
};
use crate::identity::OwnerToken;
use crate::repr::{
    constant_of, value_type_of, Bool, Idx, Representation, ScalarType, VectorElement,
    WritableRepresentation, U32,
};
use crate::schedule::AnyScalarSlot;
use crate::storage::{
    BufferViewId, BufferViewLayout, LaunchLocalId, LaunchLocalKind, LocalAllocation,
};
use crate::target::Backend;
use seismic_lang::expr::{ExprArena, NatExpr, SymbolId};
use seismic_lang::ids::{IntrinsicId, RepresentationId};
use seismic_lang::intrinsics::{AtomicOp, MathOp};
use seismic_lang::registry::{
    self, IntrinsicNumerics, IntrinsicResultType, IntrinsicUniformity, RepresentationKind,
};
use seismic_lang::types::DType;
use std::marker::PhantomData;

// ---------------------------------------------------------------------------
// Per-kernel construction state
// ---------------------------------------------------------------------------

struct BlockData<B: Backend> {
    ops: Vec<Op<B>>,
    parent: Option<BlockId>,
    /// Product of the trip counts of the enclosing repeats when every one is
    /// an arena expression; `None` when some enclosing trip count is a
    /// runtime scalar with no arena origin.
    multiplicity: Option<NatExpr>,
    /// Uniformity of the complete lexical control path reaching this block.
    control_uniformity: Uniformity,
}

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
    fn combine(self, other: Self) -> Self {
        self.max(other)
    }
}

#[derive(Clone, Copy)]
struct PlaceEntry {
    place: PlaceRef,
    representation: RepresentationId,
    rank: u32,
}

/// The construction state of the one open kernel of an implementation
/// builder. Reset by `close`.
pub(crate) struct KernelState<B: Backend> {
    owner: OwnerToken,
    kernel: u32,
    blocks: Vec<BlockData<B>>,
    values: Vec<ValueEntry>,
    places: Vec<PlaceEntry>,
    planes: Vec<(u32, u32)>,
    bindings: Vec<Binding>,
    nat_args: Vec<NatExpr>,
    scalar_args: Vec<(SymbolId, DType)>,
    result_slots: Vec<(AnyScalarSlot, DType)>,
    locals: Vec<LocalAllocation>,
    numerical: Vec<NumericalFact>,
    fact_multiplicity: Vec<Option<NatExpr>>,
    intrinsic_resources: Vec<IntrinsicResources>,
    addressable_resources: Vec<ops::AddressableResourceLease>,
    addressable_resource_cursors: Vec<NatExpr>,
    intrinsics_used: Vec<IntrinsicId>,
    barriers: u32,
    uses_subgroup: bool,
}

impl<B: Backend> KernelState<B> {
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
            planes: Vec::new(),
            bindings: Vec::new(),
            nat_args: Vec::new(),
            scalar_args: Vec::new(),
            result_slots: Vec::new(),
            locals: Vec::new(),
            numerical: Vec::new(),
            fact_multiplicity: Vec::new(),
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

pub(crate) struct Builder<'a, B: Backend> {
    expr: &'a mut ExprArena,
    views: &'a [BufferViewLayout],
    kernels: &'a mut Vec<Kernel<B>>,
    state: &'a mut KernelState<B>,
    target_facts: &'a B::Facts,
    resource_classes: &'a [crate::target::AddressableResourceClass],
    vector_support: &'a crate::target::VectorSupport,
    block: BlockId,
}

/// Opens a kernel builder over the implementation builder's tables. The
/// root block is created here; `close` pushes the kernel onto `kernels`.
pub(crate) fn open<'a, B: Backend>(
    owner: OwnerToken,
    expr: &'a mut ExprArena,
    views: &'a [BufferViewLayout],
    kernels: &'a mut Vec<Kernel<B>>,
    state: &'a mut KernelState<B>,
    target_facts: &'a B::Facts,
    resource_classes: &'a [crate::target::AddressableResourceClass],
    vector_support: &'a crate::target::VectorSupport,
) -> KernelBuilder<'a, B> {
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
    KernelBuilder {
        inner: Builder {
            expr,
            views,
            kernels,
            state,
            target_facts,
            resource_classes,
            vector_support,
            block: BlockId::new(owner, kernel, 0),
        },
    }
}

/// Core-only erased construction used by the universal semantic lowering.
/// It remains inside kernel internals: factories and backends can only use
/// the typed `KernelBuilder` surface.
#[derive(Clone, Copy, Debug)]
pub(crate) struct PortableValue {
    pub(crate) raw: ErasedValue,
    pub(crate) ty: ValueType,
}
#[derive(Clone, Copy, Debug)]
pub(crate) struct PortablePlace {
    index: u32,
    write: PortableWriteCapability,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PortableWriteCapability {
    ReadOnly,
    DenseElement,
    RepresentationPacket,
}

#[derive(Clone, Debug)]
pub(crate) struct PortableTensor {
    place: PortablePlace,
    extents: Vec<PortableValue>,
    steps: Vec<PortableViewStep>,
    plane: Option<u32>,
}

#[derive(Clone, Debug)]
enum PortableViewStep {
    Slice(Vec<PortableSliceAxis>),
    Transpose(Vec<u32>),
    Reshape {
        from: Vec<PortableValue>,
        to: Vec<PortableValue>,
    },
}

#[derive(Clone, Copy, Debug)]
pub(crate) enum PortableSliceAxis {
    Point(PortableValue),
    Range {
        start: PortableValue,
        end: PortableValue,
    },
    Full,
}

pub(crate) struct PortableBuilder<'a, B: Backend> {
    inner: Builder<'a, B>,
}

pub(crate) fn open_portable<'a, B: Backend>(
    owner: OwnerToken,
    expr: &'a mut ExprArena,
    views: &'a [BufferViewLayout],
    kernels: &'a mut Vec<Kernel<B>>,
    state: &'a mut KernelState<B>,
    target_facts: &'a B::Facts,
    resource_classes: &'a [crate::target::AddressableResourceClass],
    vector_support: &'a crate::target::VectorSupport,
) -> PortableBuilder<'a, B> {
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
            views,
            kernels,
            state,
            target_facts,
            resource_classes,
            vector_support,
            block: BlockId::new(owner, kernel, 0),
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

impl<'a, B: Backend> PortableBuilder<'a, B> {
    pub(crate) fn expression_arena(&mut self) -> &mut ExprArena {
        self.inner.expr
    }

    pub(crate) fn arg_view(
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
            index,
            write: if writable {
                PortableWriteCapability::DenseElement
            } else {
                PortableWriteCapability::ReadOnly
            },
        }
    }
    pub(crate) fn representation_destination(
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
            index,
            write: PortableWriteCapability::RepresentationPacket,
        }
    }
    pub(crate) fn local_tensor(
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
            index,
            write: PortableWriteCapability::DenseElement,
        })
    }
    pub(crate) fn nat_arg(&mut self, expr: NatExpr) -> PortableValue {
        self.nat_arg_with_ordinal(expr).0
    }
    /// Adds a compiler-owned natural argument and returns both its kernel value
    /// and ABI ordinal. Schedule specialization uses the ordinal to rebind a
    /// portable launch's logical base without changing native kernel code.
    pub(crate) fn nat_arg_with_ordinal(&mut self, expr: NatExpr) -> (PortableValue, u32) {
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
    pub(crate) fn scalar_arg(&mut self, symbol: SymbolId, dtype: DType) -> PortableValue {
        let index = self.inner.state.scalar_args.len() as u32;
        self.inner.state.scalar_args.push((symbol, dtype));
        let ty = dtype_value_type(dtype);
        let out = self
            .inner
            .define_with(ty.clone(), None, Uniformity::Workgroup);
        self.inner.emit(Op::ScalarArg { out, index });
        PortableValue { raw: out, ty }
    }
    fn geometry(&mut self, kind: GeometryValue) -> PortableValue {
        let value = self.inner.geometry(kind);
        PortableValue {
            raw: ErasedValue::new(value.owner(), value.kernel(), value.block(), value.index()),
            ty: ValueType::Index,
        }
    }
    pub(crate) fn global_id(&mut self, axis: u8) -> PortableValue {
        self.geometry(GeometryValue::GlobalId(axis))
    }
    pub(crate) fn local_id(&mut self, axis: u8) -> PortableValue {
        self.geometry(GeometryValue::LocalId(axis))
    }
    pub(crate) fn workgroup_size(&mut self, axis: u8) -> PortableValue {
        self.geometry(GeometryValue::WorkgroupSize(axis))
    }
    pub(crate) fn subgroup_lane(&mut self) -> PortableValue {
        self.inner.state.uses_subgroup = true;
        self.geometry(GeometryValue::SubgroupLane)
    }
    pub(crate) fn uniformity(
        &self,
        value: PortableValue,
    ) -> seismic_lang::registry::IntrinsicUniformity {
        match self.inner.uniformity_of(value.raw) {
            Uniformity::Workgroup => seismic_lang::registry::IntrinsicUniformity::Workgroup,
            Uniformity::Subgroup => seismic_lang::registry::IntrinsicUniformity::Subgroup,
            Uniformity::Varying => seismic_lang::registry::IntrinsicUniformity::Varying,
        }
    }
    pub(crate) fn result_slot(&mut self, slot: AnyScalarSlot) -> u32 {
        assert_eq!(
            slot.owner(),
            self.inner.owner(),
            "result slot belongs to another implementation"
        );
        let index = self.inner.state.result_slots.len() as u32;
        self.inner.state.result_slots.push((slot, slot.dtype));
        index
    }
    pub(crate) fn index_constant(&mut self, value: u64) -> PortableValue {
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
    pub(crate) fn constant(&mut self, value: ops::ConstantValue, ty: ValueType) -> PortableValue {
        let value = match (value, &ty) {
            (ops::ConstantValue::F32(value), ValueType::Scalar(DType::F16)) => {
                ops::ConstantValue::F16(crate::repr::f16_bits(value))
            }
            (ops::ConstantValue::F32(value), ValueType::Scalar(DType::BF16)) => {
                ops::ConstantValue::BF16(crate::repr::bf16_bits(value))
            }
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
    pub(crate) fn binary(
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
        let uniformity = self.inner.combined_uniformity([a.raw, b.raw]);
        let (a, b) = (self.used(a), self.used(b));
        let ty = a_type(&self.inner, a);
        let out = self.inner.define_with(ty.clone(), None, uniformity);
        self.inner.emit(Op::Binary { op, out, a, b });
        PortableValue { raw: out, ty }
    }
    pub(crate) fn unary(&mut self, op: UnaryOp, a: PortableValue) -> PortableValue {
        let ty = a.ty.clone();
        let uniformity = self.inner.uniformity_of(a.raw);
        let a = self.used(a);
        let out = self.inner.define_with(ty.clone(), None, uniformity);
        self.inner.emit(Op::Unary { op, out, a });
        PortableValue { raw: out, ty }
    }
    pub(crate) fn bit(&mut self, op: BitOp, a: PortableValue, b: PortableValue) -> PortableValue {
        assert_eq!(a.ty, b.ty, "portable bit operand types differ");
        let ty = a.ty.clone();
        let uniformity = self.inner.combined_uniformity([a.raw, b.raw]);
        let (a, b) = (self.used(a), self.used(b));
        let out = self.inner.define_with(ty.clone(), None, uniformity);
        self.inner.emit(Op::Bit { op, out, a, b });
        PortableValue { raw: out, ty }
    }
    pub(crate) fn cmp(&mut self, op: CmpOp, a: PortableValue, b: PortableValue) -> PortableValue {
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
    pub(crate) fn logic(
        &mut self,
        op: LogicOp,
        a: PortableValue,
        b: PortableValue,
    ) -> PortableValue {
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
    pub(crate) fn select(
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
    pub(crate) fn not(&mut self, a: PortableValue) -> PortableValue {
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
    pub(crate) fn cast(&mut self, a: PortableValue, to: ValueType) -> PortableValue {
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
    pub(crate) fn bitcast(&mut self, a: PortableValue, to: ValueType) -> PortableValue {
        fn width(ty: &ValueType) -> u32 {
            match ty {
                ValueType::Scalar(dtype) => dtype.bytes(),
                ValueType::Vector { dtype, lanes } => dtype
                    .bytes()
                    .checked_mul(u32::from(*lanes))
                    .expect("vector bit width overflow"),
                ValueType::Index => 8,
                ValueType::Bool => 1,
                ValueType::Opaque { .. } => panic!("opaque values cannot be bitcast"),
            }
        }
        assert_eq!(
            width(&a.ty),
            width(&to),
            "bitcast requires equal-width value types"
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
    pub(crate) fn math(&mut self, op: MathOp, a: PortableValue) -> PortableValue {
        super::reference_math::expand(self, op, a)
    }
    pub(crate) fn math_approximate(&mut self, op: MathOp, a: PortableValue) -> PortableValue {
        let ty = a.ty;
        let uniformity = self.inner.uniformity_of(a.raw);
        let a = self.used(a);
        let out = self.inner.define_with(ty, None, uniformity);
        self.inner.record_fact(NumericalFact::ApproximateMath(op));
        self.inner.emit(Op::Math {
            op,
            precision: MathPrecision::Approximate,
            out,
            a,
        });
        PortableValue { raw: out, ty }
    }
    pub(crate) fn fma(
        &mut self,
        a: PortableValue,
        b: PortableValue,
        c: PortableValue,
    ) -> PortableValue {
        assert_eq!(a.ty, b.ty);
        assert_eq!(a.ty, c.ty);
        let ty = a.ty.clone();
        let uniformity = self.inner.combined_uniformity([a.raw, b.raw, c.raw]);
        let (a, b, c) = (self.used(a), self.used(b), self.used(c));
        let out = self.inner.define_with(ty.clone(), None, uniformity);
        self.inner.emit(Op::Fma { out, a, b, c });
        PortableValue { raw: out, ty }
    }
    pub(crate) fn read(&mut self, place: PortablePlace, index: &[PortableValue]) -> PortableValue {
        let entry = *self
            .inner
            .state
            .places
            .get(place.index as usize)
            .expect("portable place belongs to the open kernel");
        assert_eq!(
            entry.rank as usize,
            index.len(),
            "portable read rank mismatch"
        );
        let uniformity = self
            .inner
            .combined_uniformity(index.iter().map(|value| value.raw))
            .combine(self.inner.state.blocks[self.inner.block.index() as usize].control_uniformity);
        let indices = index
            .iter()
            .map(|value| {
                assert_eq!(value.ty, ValueType::Index);
                self.inner.use_value(value.raw, &ValueType::Index)
            })
            .collect();
        let ty = match registry::representation_info(entry.representation).kind {
            RepresentationKind::Dense(dtype) => dtype_value_type(dtype),
            RepresentationKind::Packed(_) => ValueType::Scalar(DType::F32),
            RepresentationKind::External(_) => {
                panic!("external packets are readable only through a registered conversion")
            }
        };
        // A deterministic read at the same address on every participant is
        // uniform. Preserve the index/control uniformity instead of treating
        // every memory read as varying; this is what permits a scalar tensor
        // lookup to cross a sequential kernel/schedule cut.
        let out = self.inner.define_with(ty.clone(), None, uniformity);
        self.inner.emit(Op::Read {
            out,
            place: entry.place,
            representation: entry.representation,
            index: indices,
        });
        PortableValue { raw: out, ty }
    }
    pub(crate) fn write(
        &mut self,
        place: PortablePlace,
        index: &[PortableValue],
        value: PortableValue,
    ) {
        assert_eq!(
            place.write,
            PortableWriteCapability::DenseElement,
            "portable write requires dense-element write authority"
        );
        let entry = *self
            .inner
            .state
            .places
            .get(place.index as usize)
            .expect("portable place belongs to the open kernel");
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
    pub(crate) fn atomic(
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
        let entry = *self
            .inner
            .state
            .places
            .get(place.index as usize)
            .expect("portable place belongs to the open kernel");
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
    pub(crate) fn extent(&mut self, place: PortablePlace, axis: u32) -> PortableValue {
        let entry = *self
            .inner
            .state
            .places
            .get(place.index as usize)
            .expect("portable place belongs to the open kernel");
        assert!(
            axis < entry.rank,
            "portable extent axis is outside the place rank"
        );
        let nat = match entry.place {
            PlaceRef::Global { slot } => {
                let view = self.inner.state.bindings[slot.index() as usize].view;
                Some(self.inner.views[view.index as usize].extents[axis as usize])
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
    pub(crate) fn tensor(&mut self, place: PortablePlace) -> PortableTensor {
        let rank = self.inner.state.places[place.index as usize].rank;
        let extents = (0..rank).map(|axis| self.extent(place, axis)).collect();
        PortableTensor {
            place,
            extents,
            steps: Vec::new(),
            plane: None,
        }
    }
    pub(crate) fn tensor_extents<'b>(&self, tensor: &'b PortableTensor) -> &'b [PortableValue] {
        &tensor.extents
    }
    pub(crate) fn representation_convert_packet(
        &mut self,
        source: &PortableTensor,
        destination: &PortableTensor,
        conversion: seismic_lang::ids::RepresentationConversionId,
        packet: PortableValue,
    ) {
        let recipe = registry::representation_conversion_info(conversion);
        let source_place = self.inner.state.places[source.place.index as usize];
        let destination_place = self.inner.state.places[destination.place.index as usize];
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
            source.steps.is_empty() && source.plane.is_none(),
            "external conversion source must be the canonical complete packet view"
        );
        assert!(
            destination.steps.is_empty() && destination.plane.is_none(),
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
        let entry = self.inner.state.places[tensor.place.index as usize];
        let steps = tensor
            .steps
            .iter()
            .map(|step| match step {
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
        ops::LogicalTensorMap {
            base: entry.place,
            representation: entry.representation,
            logical_extents: tensor
                .extents
                .iter()
                .map(|value| {
                    self.inner
                        .nat_of(value.raw)
                        .expect("logical tensor extent has no canonical NatExpr")
                })
                .collect(),
            extents: tensor.extents.iter().map(|value| value.raw).collect(),
            steps,
        }
    }
    pub(crate) fn tensor_plane(&self, mut tensor: PortableTensor, plane: u32) -> PortableTensor {
        assert!(
            tensor.plane.replace(plane).is_none(),
            "plane view was applied twice"
        );
        tensor
    }
    pub(crate) fn tensor_slice(
        &mut self,
        mut tensor: PortableTensor,
        axes: Vec<PortableSliceAxis>,
    ) -> PortableTensor {
        assert_eq!(axes.len(), tensor.extents.len(), "slice rank mismatch");
        let mut extents = Vec::new();
        for (axis, extent) in axes.iter().zip(&tensor.extents) {
            match *axis {
                PortableSliceAxis::Point(_) => {}
                PortableSliceAxis::Range { start, end } => {
                    extents.push(self.binary(BinaryOp::Sub, end, start));
                }
                PortableSliceAxis::Full => extents.push(*extent),
            }
        }
        tensor.steps.push(PortableViewStep::Slice(axes));
        tensor.extents = extents;
        tensor
    }
    pub(crate) fn tensor_transpose(
        &self,
        mut tensor: PortableTensor,
        permutation: Vec<u32>,
    ) -> PortableTensor {
        assert_eq!(
            permutation.len(),
            tensor.extents.len(),
            "transpose rank mismatch"
        );
        let mut seen = vec![false; permutation.len()];
        let extents = permutation
            .iter()
            .map(|axis| {
                let axis = *axis as usize;
                assert!(
                    axis < seen.len() && !seen[axis],
                    "invalid transpose permutation"
                );
                seen[axis] = true;
                tensor.extents[axis]
            })
            .collect();
        tensor.steps.push(PortableViewStep::Transpose(permutation));
        tensor.extents = extents;
        tensor
    }
    pub(crate) fn tensor_reshape(
        &self,
        mut tensor: PortableTensor,
        extents: Vec<PortableValue>,
    ) -> PortableTensor {
        assert!(extents.iter().all(|value| value.ty == ValueType::Index));
        tensor.steps.push(PortableViewStep::Reshape {
            from: tensor.extents.clone(),
            to: extents.clone(),
        });
        tensor.extents = extents;
        tensor
    }
    fn tensor_indices(
        &mut self,
        tensor: &PortableTensor,
        index: &[PortableValue],
    ) -> Vec<PortableValue> {
        assert_eq!(
            index.len(),
            tensor.extents.len(),
            "logical tensor index rank mismatch"
        );
        let mut index = index.to_vec();
        for step in tensor.steps.iter().rev() {
            index = match step {
                PortableViewStep::Slice(axes) => {
                    let mut output = Vec::with_capacity(axes.len());
                    let mut values = index.iter().copied();
                    for axis in axes {
                        output.push(match *axis {
                            PortableSliceAxis::Point(value) => value,
                            PortableSliceAxis::Range { start, .. } => self.binary(
                                BinaryOp::Add,
                                start,
                                values.next().expect("slice mapping rank is closed"),
                            ),
                            PortableSliceAxis::Full => {
                                values.next().expect("slice mapping rank is closed")
                            }
                        });
                    }
                    assert!(values.next().is_none(), "slice mapping left an output axis");
                    output
                }
                PortableViewStep::Transpose(permutation) => {
                    let zero = self.index_constant(0);
                    let mut output = vec![zero; permutation.len()];
                    for (axis, source) in permutation.iter().zip(&index) {
                        output[*axis as usize] = *source;
                    }
                    output
                }
                PortableViewStep::Reshape { from, to } => {
                    let mut linear = self.index_constant(0);
                    for (coordinate, extent) in index.iter().zip(to) {
                        linear = self.binary(BinaryOp::Mul, linear, *extent);
                        linear = self.binary(BinaryOp::Add, linear, *coordinate);
                    }
                    let zero = self.index_constant(0);
                    let mut output = vec![zero; from.len()];
                    for axis in (0..from.len()).rev() {
                        output[axis] = self.binary(BinaryOp::Rem, linear, from[axis]);
                        linear = self.binary(BinaryOp::Div, linear, from[axis]);
                    }
                    output
                }
            };
        }
        index
    }
    pub(crate) fn tensor_read(
        &mut self,
        tensor: &PortableTensor,
        index: &[PortableValue],
    ) -> PortableValue {
        let index = self.tensor_indices(tensor, index);
        match tensor.plane {
            None => self.read(tensor.place, &index),
            Some(plane) => {
                let entry = *self
                    .inner
                    .state
                    .places
                    .get(tensor.place.index as usize)
                    .expect("portable tensor belongs to the open kernel");
                let RepresentationKind::Packed(layout) =
                    &registry::representation_info(entry.representation).kind
                else {
                    panic!("plane reads require a packed representation")
                };
                assert!(
                    (plane as usize) < layout.planes.len(),
                    "plane index is outside the packed representation"
                );
                let indices = index
                    .iter()
                    .map(|value| {
                        assert_eq!(value.ty, ValueType::Index);
                        self.inner.use_value(value.raw, &ValueType::Index)
                    })
                    .collect();
                let ty = ValueType::Scalar(DType::U32);
                let out = self.inner.define(ty.clone());
                self.inner.emit(Op::ReadPlane {
                    out,
                    place: entry.place,
                    plane,
                    index: indices,
                });
                PortableValue { raw: out, ty }
            }
        }
    }
    pub(crate) fn tensor_write(
        &mut self,
        tensor: &PortableTensor,
        index: &[PortableValue],
        value: PortableValue,
    ) {
        assert!(tensor.plane.is_none(), "plane views are read-only");
        let index = self.tensor_indices(tensor, index);
        self.write(tensor.place, &index, value);
    }
    pub(crate) fn tensor_atomic(
        &mut self,
        op: AtomicOp,
        tensor: &PortableTensor,
        index: &[PortableValue],
        value: PortableValue,
    ) {
        assert!(
            tensor.plane.is_none(),
            "plane views are not atomic destinations"
        );
        let index = self.tensor_indices(tensor, index);
        self.atomic(op, tensor.place, &index, value);
    }
    /// Publishes one lane's failed preflight predicate into the canonical
    /// dense-u32 status word. Atomic max is an order-independent all-lanes OR
    /// and needs no barrier or elected writer.
    pub(crate) fn preflight_fail(&mut self, status: &PortableTensor, condition: PortableValue) {
        assert_eq!(
            condition.ty,
            ValueType::Bool,
            "preflight predicate must be boolean"
        );
        assert_eq!(
            status.extents.len(),
            1,
            "preflight status is one-dimensional"
        );
        assert!(
            status.steps.is_empty() && status.plane.is_none(),
            "preflight status is an identity dense view"
        );
        let entry = self.inner.state.places[status.place.index as usize];
        assert_eq!(
            entry.representation,
            registry::dense(DType::U32),
            "preflight status uses dense u32 storage"
        );
        let failed = self.not(condition);
        let failed = self.cast(failed, ValueType::Scalar(DType::U32));
        let zero = self.index_constant(0);
        self.tensor_atomic(AtomicOp::Max, status, &[zero], failed);
    }
    pub(crate) fn store_slot(&mut self, slot: u32, value: PortableValue) {
        let expected = self
            .inner
            .state
            .result_slots
            .get(slot as usize)
            .expect("portable result slot belongs to the open kernel")
            .1;
        assert_eq!(
            value.ty,
            dtype_value_type(expected),
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
    pub(crate) fn repeat(
        &mut self,
        start: PortableValue,
        end: PortableValue,
        initial: Vec<PortableValue>,
        body: impl FnOnce(
            &mut PortableBuilder<'_, B>,
            PortableValue,
            Vec<PortableValue>,
        ) -> Vec<PortableValue>,
    ) -> Vec<PortableValue> {
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
        let block = self.inner.new_block(multiplicity, control_uniformity);
        let (binder, carry_params, next) = {
            let mut sub = self.inner.in_block(block);
            let binder = sub.define_with(ValueType::Index, None, range_uniformity);
            let params = carries_in
                .iter()
                .zip(schema.values())
                .map(|(initial, ty)| {
                    let uniformity = sub.uniformity_of(*initial).combine(control_uniformity);
                    sub.define_with(ty.clone(), None, uniformity)
                })
                .collect::<Vec<_>>();
            let carried = params
                .iter()
                .copied()
                .zip(schema.values())
                .map(|(raw, ty)| PortableValue {
                    raw,
                    ty: ty.clone(),
                })
                .collect();
            let mut builder = PortableBuilder { inner: sub };
            let next = body(
                &mut builder,
                PortableValue {
                    raw: binder,
                    ty: ValueType::Index,
                },
                carried,
            );
            assert_eq!(
                next.len(),
                schema.len(),
                "portable repeat body result count differs from its carry schema"
            );
            (binder, params, next)
        };
        let mut sub = self.inner.in_block(block);
        let yielded = next
            .iter()
            .zip(schema.values())
            .map(|(value, ty)| {
                assert_eq!(&value.ty, ty);
                sub.use_value(value.raw, ty)
            })
            .collect::<Vec<_>>();
        sub.emit(Op::Yield { values: yielded });
        drop(sub);
        let outs = next
            .iter()
            .zip(schema.values())
            .map(|(value, ty)| {
                let uniformity = self
                    .inner
                    .uniformity_of(value.raw)
                    .combine(control_uniformity);
                self.inner.define_with(ty.clone(), None, uniformity)
            })
            .collect::<Vec<_>>();
        self.inner.emit(Op::Repeat {
            start: start_raw,
            end: end_raw,
            binder,
            carries_in,
            carry_params,
            body: block,
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
    pub(crate) fn branch(
        &mut self,
        condition: PortableValue,
        then: impl FnOnce(&mut PortableBuilder<'_, B>) -> Vec<PortableValue>,
        otherwise: impl FnOnce(&mut PortableBuilder<'_, B>) -> Vec<PortableValue>,
    ) -> Vec<PortableValue> {
        assert_eq!(condition.ty, ValueType::Bool);
        let condition = self.used(condition);
        let multiplicity = self.inner.state.blocks[self.inner.block.index() as usize].multiplicity;
        let control_uniformity = self.inner.state.blocks[self.inner.block.index() as usize]
            .control_uniformity
            .combine(self.inner.uniformity_of(condition));
        let then_block = self.inner.new_block(multiplicity, control_uniformity);
        let then_values = {
            let mut child = PortableBuilder {
                inner: self.inner.in_block(then_block),
            };
            then(&mut child)
        };
        let schema = ValueSchema::new(then_values.iter().map(|value| value.ty.clone()).collect());
        self.inner.yield_values(
            then_block,
            &then_values
                .iter()
                .map(|value| value.raw)
                .collect::<Vec<_>>(),
            &schema,
        );
        let else_block = self.inner.new_block(multiplicity, control_uniformity);
        let else_values = {
            let mut child = PortableBuilder {
                inner: self.inner.in_block(else_block),
            };
            otherwise(&mut child)
        };
        assert_eq!(
            else_values
                .iter()
                .map(|value| &value.ty)
                .collect::<Vec<_>>(),
            schema.values().iter().collect::<Vec<_>>(),
            "portable branch result schemas differ"
        );
        self.inner.yield_values(
            else_block,
            &else_values
                .iter()
                .map(|value| value.raw)
                .collect::<Vec<_>>(),
            &schema,
        );
        let outs = then_values
            .iter()
            .zip(else_values.iter())
            .zip(schema.values())
            .map(|((a, b), ty)| {
                let uniformity = control_uniformity
                    .combine(self.inner.uniformity_of(a.raw))
                    .combine(self.inner.uniformity_of(b.raw));
                self.inner.define_with(ty.clone(), None, uniformity)
            })
            .collect::<Vec<_>>();
        self.inner.emit(Op::Branch {
            cond: condition,
            then: then_block,
            otherwise: else_block,
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
    pub(crate) fn close(self) -> KernelId {
        self.inner.close()
    }
}

fn a_type<B: Backend>(builder: &Builder<'_, B>, value: ErasedValue) -> ValueType {
    builder.entry(value).ty.clone()
}

impl<'a, B: Backend> Builder<'a, B> {
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
            views: self.views,
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

    fn define(&mut self, ty: ValueType) -> ErasedValue {
        self.define_with(ty, None, Uniformity::Varying)
    }

    fn define_nat(&mut self, ty: ValueType, nat: Option<NatExpr>) -> ErasedValue {
        self.define_with(ty, nat, Uniformity::Varying)
    }

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

    fn use_scalar<T: ScalarType>(&mut self, value: ScalarId<T>) -> ErasedValue {
        self.assert_kernel(value.owner(), value.kernel());
        assert_eq!(
            value.block(),
            self.entry(ErasedValue::new(
                value.owner(),
                value.kernel(),
                value.block(),
                value.index()
            ))
            .block,
            "typed value carries the wrong defining block"
        );
        self.use_value(
            ErasedValue::new(value.owner(), value.kernel(), value.block(), value.index()),
            &value_type_of::<T>(),
        )
    }

    fn use_vector<T: VectorElement, const LANES: u16>(
        &mut self,
        value: VectorId<T, LANES>,
    ) -> ErasedValue {
        self.assert_kernel(value.owner, value.kernel);
        assert_eq!(
            value.block,
            self.entry(value.erased()).block,
            "typed vector carries the wrong defining block"
        );
        self.use_value(value.erased(), &VectorId::<T, LANES>::value_type())
    }

    fn use_index_list(&mut self, index: &[ScalarId<Idx>]) -> Vec<ErasedValue> {
        index.iter().map(|i| self.use_scalar(*i)).collect()
    }

    fn nat_of(&self, value: ErasedValue) -> Option<NatExpr> {
        self.entry(value).nat
    }

    fn uniformity_of(&self, value: ErasedValue) -> Uniformity {
        self.entry(value).uniformity
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

    fn scalar_out<T: ScalarType>(&mut self) -> (ErasedValue, ScalarId<T>) {
        self.scalar_out_with(Uniformity::Varying)
    }

    fn scalar_out_with<T: ScalarType>(
        &mut self,
        uniformity: Uniformity,
    ) -> (ErasedValue, ScalarId<T>) {
        let out = self.define_with(value_type_of::<T>(), None, uniformity);
        (
            out,
            ScalarId::new(out.owner, out.kernel, out.block, out.index()),
        )
    }

    fn vector_out<T: VectorElement, const LANES: u16>(
        &mut self,
        uniformity: Uniformity,
    ) -> (ErasedValue, VectorId<T, LANES>) {
        let out = self.define_with(VectorId::<T, LANES>::value_type(), None, uniformity);
        (
            out,
            VectorId::new(out.owner, out.kernel, out.block, out.index()),
        )
    }

    fn index_out(&mut self, nat: Option<NatExpr>) -> (ErasedValue, ScalarId<Idx>) {
        self.index_out_with(nat, Uniformity::Varying)
    }

    fn index_out_with(
        &mut self,
        nat: Option<NatExpr>,
        uniformity: Uniformity,
    ) -> (ErasedValue, ScalarId<Idx>) {
        let out = self.define_with(ValueType::Index, nat, uniformity);
        (
            out,
            ScalarId::new(out.owner, out.kernel, out.block, out.index()),
        )
    }

    // ----- places ------------------------------------------------------------

    fn place(&self, owner: OwnerToken, kernel: u32, block: BlockId, index: u32) -> PlaceEntry {
        self.assert_kernel(owner, kernel);
        if !self.dominates(block, self.block) {
            panic!(
                "kernel builder: place #{index} defined in {block:?} is not visible in {:?}",
                self.block
            );
        }
        match self.state.places.get(index as usize) {
            Some(place) => *place,
            None => panic!("kernel builder: place #{index} is not a place of the open kernel"),
        }
    }

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
        let extents = match self.views.get(view.index as usize) {
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

    fn checked_index(&mut self, place: PlaceEntry, index: &[ScalarId<Idx>]) -> Vec<ErasedValue> {
        if index.len() as u32 != place.rank {
            panic!(
                "kernel builder: access with {} indices to a place of rank {}",
                index.len(),
                place.rank
            );
        }
        self.use_index_list(index)
    }

    fn record_fact(&mut self, fact: NumericalFact) {
        let multiplicity = self.state.blocks[self.block.index() as usize].multiplicity;
        self.state.numerical.push(fact);
        self.state.fact_multiplicity.push(multiplicity);
    }

    fn record_intrinsic_numerics(
        &mut self,
        signature: &seismic_lang::registry::IntrinsicSignature,
        op: &B::Intrinsic,
    ) {
        let semantics = B::intrinsic_numerics(self.target_facts, signature, op);
        match semantics.arithmetic {
            IntrinsicNumerics::Exact => {}
            IntrinsicNumerics::Reassociated { accumulator } => {
                self.record_fact(NumericalFact::ReassociatedIntrinsic(signature.id));
                if accumulator.is_float() && accumulator != DType::F32 {
                    self.record_fact(NumericalFact::NarrowAccumulator(accumulator));
                }
            }
            IntrinsicNumerics::Approximate { .. } | IntrinsicNumerics::Unknown => {
                self.record_fact(NumericalFact::ReassociatedIntrinsic(signature.id));
            }
        }
        if semantics.flush_to_zero {
            self.record_fact(NumericalFact::FlushToZero);
        }
    }

    // ----- public-method servers ---------------------------------------------

    pub(super) fn arg_readable<R: Representation>(
        &mut self,
        v: BufferViewId<R>,
    ) -> ReadablePlaceId<R> {
        let (slot, rank) = self.bind_view(v.erase(), BindingAccess::Read);
        let index = self.push_place(PlaceRef::Global { slot }, R::id(), rank);
        ReadablePlaceId::new(self.owner(), self.kernel_index(), self.block, index)
    }
    pub(super) fn arg_writable<R: WritableRepresentation>(
        &mut self,
        v: BufferViewId<R>,
    ) -> WritablePlaceId<R> {
        let (slot, rank) = self.bind_view(v.erase(), BindingAccess::Write);
        let index = self.push_place(PlaceRef::Global { slot }, R::id(), rank);
        WritablePlaceId::new(self.owner(), self.kernel_index(), self.block, index)
    }
    pub(super) fn arg_nat(&mut self, e: NatExpr) -> ScalarId<Idx> {
        let index = self.state.nat_args.len() as u32;
        self.state.nat_args.push(e);
        let (out, id) = self.index_out_with(Some(e), Uniformity::Workgroup);
        self.emit(Op::NatArg { out, index });
        id
    }
    pub(super) fn arg_scalar<T: ScalarType>(&mut self, s: SymbolId) -> ScalarId<T> {
        let index = self.state.scalar_args.len() as u32;
        self.state.scalar_args.push((s, T::DTYPE));
        let (out, id) = self.scalar_out_with::<T>(Uniformity::Workgroup);
        self.emit(Op::ScalarArg { out, index });
        id
    }
    pub(super) fn result_slot<T: ScalarType>(
        &mut self,
        s: crate::schedule::ScalarSlotId<T>,
    ) -> WritableScalar<T> {
        assert_eq!(
            s.erase().owner(),
            self.owner(),
            "result slot belongs to another implementation"
        );
        let index = self.state.result_slots.len() as u32;
        self.state.result_slots.push((s.erase(), T::DTYPE));
        WritableScalar {
            owner: self.owner(),
            kernel: self.kernel_index(),
            index,
            marker: PhantomData,
        }
    }
    pub(super) fn local<R: Representation>(
        &mut self,
        k: LaunchLocalKind,
        e: Vec<NatExpr>,
    ) -> LaunchLocalId<R> {
        let index = self.state.locals.len() as u32;
        self.state.locals.push(LocalAllocation {
            kind: k,
            representation: R::id(),
            extents: e,
            alignment: representation_alignment(R::id()),
        });
        LaunchLocalId::new(self.owner(), self.kernel_index(), index)
    }
    fn local_place<R: Representation>(&mut self, l: LaunchLocalId<R>) -> u32 {
        self.assert_kernel(l.owner(), l.kernel());
        let rank = self.state.locals[l.index() as usize].extents.len() as u32;
        self.push_place(PlaceRef::Local { index: l.index() }, R::id(), rank)
    }
    pub(super) fn local_readable<R: Representation>(
        &mut self,
        l: LaunchLocalId<R>,
    ) -> ReadablePlaceId<R> {
        let index = self.local_place(l);
        ReadablePlaceId::new(self.owner(), self.kernel_index(), self.block, index)
    }
    pub(super) fn local_writable<R: WritableRepresentation>(
        &mut self,
        l: LaunchLocalId<R>,
    ) -> WritablePlaceId<R> {
        let index = self.local_place(l);
        WritablePlaceId::new(self.owner(), self.kernel_index(), self.block, index)
    }
    pub(super) fn as_readable<R: Representation>(
        &mut self,
        p: WritablePlaceId<R>,
    ) -> ReadablePlaceId<R> {
        let _ = self.place(p.owner(), p.kernel(), p.block(), p.index());
        ReadablePlaceId::new(p.owner(), p.kernel(), p.block(), p.index())
    }

    fn geometry(&mut self, kind: GeometryValue) -> ScalarId<Idx> {
        let uniformity = match kind {
            GeometryValue::WorkgroupId(_)
            | GeometryValue::WorkgroupSize(_)
            | GeometryValue::GridSize(_) => Uniformity::Workgroup,
            GeometryValue::LocalId(_)
            | GeometryValue::GlobalId(_)
            | GeometryValue::SubgroupLane => Uniformity::Varying,
        };
        let (out, id) = self.index_out_with(None, uniformity);
        self.emit(Op::Geometry { out, kind });
        id
    }
    pub(super) fn workgroup_id(&mut self, a: u8) -> ScalarId<Idx> {
        self.geometry(GeometryValue::WorkgroupId(a))
    }
    pub(super) fn local_id(&mut self, a: u8) -> ScalarId<Idx> {
        self.geometry(GeometryValue::LocalId(a))
    }
    pub(super) fn global_id(&mut self, a: u8) -> ScalarId<Idx> {
        self.geometry(GeometryValue::GlobalId(a))
    }
    pub(super) fn workgroup_size(&mut self, a: u8) -> ScalarId<Idx> {
        self.geometry(GeometryValue::WorkgroupSize(a))
    }
    pub(super) fn grid_size(&mut self, a: u8) -> ScalarId<Idx> {
        self.geometry(GeometryValue::GridSize(a))
    }
    pub(super) fn subgroup_lane(&mut self) -> ScalarId<Idx> {
        self.state.uses_subgroup = true;
        self.geometry(GeometryValue::SubgroupLane)
    }

    pub(super) fn constant<T: ScalarType>(&mut self, v: T::Value) -> ScalarId<T> {
        let value = constant_of::<T>(v);
        let nat = match value {
            ops::ConstantValue::Index(n) => Some(self.expr.nat(n)),
            _ => None,
        };
        let out = self.define_with(value_type_of::<T>(), nat, Uniformity::Workgroup);
        self.emit(Op::Constant { out, value });
        ScalarId::new(out.owner, out.kernel, out.block, out.index())
    }
    pub(super) fn binary<T: ScalarType>(
        &mut self,
        op: BinaryOp,
        a: ScalarId<T>,
        b: ScalarId<T>,
    ) -> ScalarId<T> {
        let (a, b) = (self.use_scalar(a), self.use_scalar(b));
        let nat = match (self.nat_of(a), self.nat_of(b)) {
            (Some(x), Some(y)) if value_type_of::<T>() == ValueType::Index => Some(match op {
                BinaryOp::Add => self.expr.nat_add(x, y),
                BinaryOp::Sub => {
                    let m = self.expr.nat_max(x, y);
                    self.expr.nat_sub(m, y)
                }
                BinaryOp::Mul => self.expr.nat_mul(x, y),
                BinaryOp::Div => self.expr.nat_div(x, y),
                BinaryOp::Rem => self.expr.nat_rem(x, y),
                BinaryOp::Min => self.expr.nat_min(x, y),
                BinaryOp::Max => self.expr.nat_max(x, y),
            }),
            _ => None,
        };
        let uniformity = self.uniformity_of(a).combine(self.uniformity_of(b));
        let out = self.define_with(value_type_of::<T>(), nat, uniformity);
        self.emit(Op::Binary { op, out, a, b });
        ScalarId::new(out.owner, out.kernel, out.block, out.index())
    }
    pub(super) fn unary<T: ScalarType>(&mut self, op: UnaryOp, a: ScalarId<T>) -> ScalarId<T> {
        let a = self.use_scalar(a);
        let (out, id) = self.scalar_out_with::<T>(self.uniformity_of(a));
        self.emit(Op::Unary { op, out, a });
        id
    }
    pub(super) fn bit<T: ScalarType>(
        &mut self,
        op: BitOp,
        a: ScalarId<T>,
        b: ScalarId<T>,
    ) -> ScalarId<T> {
        let (a, b) = (self.use_scalar(a), self.use_scalar(b));
        let (out, id) = self.scalar_out_with::<T>(self.combined_uniformity([a, b]));
        self.emit(Op::Bit { op, out, a, b });
        id
    }
    pub(super) fn fma<T: ScalarType>(
        &mut self,
        a: ScalarId<T>,
        b: ScalarId<T>,
        c: ScalarId<T>,
        contracted: bool,
    ) -> ScalarId<T> {
        let (a, b, c) = (self.use_scalar(a), self.use_scalar(b), self.use_scalar(c));
        let (out, id) = self.scalar_out_with::<T>(self.combined_uniformity([a, b, c]));
        if contracted {
            self.record_fact(NumericalFact::ContractedFma);
        }
        self.emit(Op::Fma { out, a, b, c });
        id
    }
    pub(super) fn math<T: ScalarType>(
        &mut self,
        op: MathOp,
        a: ScalarId<T>,
        p: MathPrecision,
    ) -> ScalarId<T> {
        if p == MathPrecision::Exact {
            let a = self.use_scalar(a);
            let mut portable = PortableBuilder {
                inner: self.reborrow(),
            };
            let out = super::reference_math::expand(
                &mut portable,
                op,
                PortableValue {
                    raw: a,
                    ty: value_type_of::<T>(),
                },
            );
            return ScalarId::new(
                out.raw.owner,
                out.raw.kernel,
                out.raw.block,
                out.raw.index(),
            );
        }
        let a = self.use_scalar(a);
        let (out, id) = self.scalar_out_with::<T>(self.uniformity_of(a));
        if p == MathPrecision::Approximate {
            self.record_fact(NumericalFact::ApproximateMath(op));
        }
        self.emit(Op::Math {
            op,
            precision: p,
            out,
            a,
        });
        id
    }
    pub(super) fn cast<F: ScalarType, T: ScalarType>(&mut self, a: ScalarId<F>) -> ScalarId<T> {
        let a = self.use_scalar(a);
        let to = value_type_of::<T>();
        let (out, id) = self.scalar_out_with::<T>(self.uniformity_of(a));
        self.emit(Op::Cast { out, a, to });
        id
    }
    pub(super) fn cmp<T: ScalarType>(
        &mut self,
        op: CmpOp,
        a: ScalarId<T>,
        b: ScalarId<T>,
    ) -> ScalarId<Bool> {
        let (a, b) = (self.use_scalar(a), self.use_scalar(b));
        let (out, id) = self.scalar_out_with::<Bool>(self.combined_uniformity([a, b]));
        self.emit(Op::Cmp { op, out, a, b });
        id
    }
    pub(super) fn select<T: ScalarType>(
        &mut self,
        c: ScalarId<Bool>,
        a: ScalarId<T>,
        b: ScalarId<T>,
    ) -> ScalarId<T> {
        let cond = self.use_scalar(c);
        let (a, b) = (self.use_scalar(a), self.use_scalar(b));
        let (out, id) = self.scalar_out_with::<T>(self.combined_uniformity([cond, a, b]));
        self.emit(Op::Select { out, cond, a, b });
        id
    }
    pub(super) fn logic(
        &mut self,
        op: LogicOp,
        a: ScalarId<Bool>,
        b: ScalarId<Bool>,
    ) -> ScalarId<Bool> {
        let (a, b) = (self.use_scalar(a), self.use_scalar(b));
        let (out, id) = self.scalar_out_with::<Bool>(self.combined_uniformity([a, b]));
        self.emit(Op::Logic { op, out, a, b });
        id
    }
    pub(super) fn not(&mut self, a: ScalarId<Bool>) -> ScalarId<Bool> {
        let a = self.use_scalar(a);
        let (out, id) = self.scalar_out_with::<Bool>(self.uniformity_of(a));
        self.emit(Op::Not { out, a });
        id
    }
    fn require_vector<T: VectorElement, const LANES: u16>(
        &self,
        operation: crate::target::VectorOperationClass,
    ) {
        assert!(
            self.vector_support.supports(T::DTYPE, LANES, operation),
            "ImplementationFactory constructed a vector operation absent from DeviceContract vector support"
        );
    }
    pub(super) fn vector_splat<T: VectorElement, const LANES: u16>(
        &mut self,
        value: ScalarId<T>,
    ) -> VectorId<T, LANES> {
        self.require_vector::<T, LANES>(crate::target::VectorOperationClass::Splat);
        let value = self.use_scalar(value);
        let (out, id) = self.vector_out::<T, LANES>(self.uniformity_of(value));
        self.emit(Op::VectorSplat { out, value });
        id
    }
    pub(super) fn vector_binary<T: VectorElement, const LANES: u16>(
        &mut self,
        op: BinaryOp,
        a: VectorId<T, LANES>,
        b: VectorId<T, LANES>,
    ) -> VectorId<T, LANES> {
        self.require_vector::<T, LANES>(crate::target::VectorOperationClass::Binary(op));
        let (a, b) = (self.use_vector(a), self.use_vector(b));
        let (out, id) = self.vector_out::<T, LANES>(self.combined_uniformity([a, b]));
        self.emit(Op::VectorBinary { op, out, a, b });
        id
    }
    pub(super) fn vector_unary<T: VectorElement, const LANES: u16>(
        &mut self,
        op: UnaryOp,
        value: VectorId<T, LANES>,
    ) -> VectorId<T, LANES> {
        self.require_vector::<T, LANES>(crate::target::VectorOperationClass::Unary(op));
        let value = self.use_vector(value);
        let (out, id) = self.vector_out::<T, LANES>(self.uniformity_of(value));
        self.emit(Op::VectorUnary { op, out, a: value });
        id
    }
    pub(super) fn vector_bit<T: VectorElement, const LANES: u16>(
        &mut self,
        op: BitOp,
        a: VectorId<T, LANES>,
        b: VectorId<T, LANES>,
    ) -> VectorId<T, LANES> {
        self.require_vector::<T, LANES>(crate::target::VectorOperationClass::Bit(op));
        let (a, b) = (self.use_vector(a), self.use_vector(b));
        let (out, id) = self.vector_out::<T, LANES>(self.combined_uniformity([a, b]));
        self.emit(Op::VectorBit { op, out, a, b });
        id
    }
    pub(super) fn vector_fma<T: VectorElement, const LANES: u16>(
        &mut self,
        a: VectorId<T, LANES>,
        b: VectorId<T, LANES>,
        c: VectorId<T, LANES>,
    ) -> VectorId<T, LANES> {
        self.require_vector::<T, LANES>(crate::target::VectorOperationClass::Fma);
        let (a, b, c) = (self.use_vector(a), self.use_vector(b), self.use_vector(c));
        let (out, id) = self.vector_out::<T, LANES>(self.combined_uniformity([a, b, c]));
        self.emit(Op::VectorFma { out, a, b, c });
        id
    }
    pub(super) fn vector_cast<From: VectorElement, To: VectorElement, const LANES: u16>(
        &mut self,
        value: VectorId<From, LANES>,
    ) -> VectorId<To, LANES> {
        self.require_vector::<From, LANES>(crate::target::VectorOperationClass::Cast {
            to: To::DTYPE,
        });
        let value = self.use_vector(value);
        let to = VectorId::<To, LANES>::value_type();
        let (out, id) = self.vector_out::<To, LANES>(self.uniformity_of(value));
        self.emit(Op::VectorCast { out, a: value, to });
        id
    }
    pub(super) fn vector_lane<T: VectorElement, const LANES: u16>(
        &mut self,
        vector: VectorId<T, LANES>,
        lane: u16,
    ) -> ScalarId<T> {
        self.require_vector::<T, LANES>(crate::target::VectorOperationClass::Lane);
        assert!(lane < LANES, "vector lane is outside its fixed width");
        let vector = self.use_vector(vector);
        let (out, id) = self.scalar_out_with::<T>(self.uniformity_of(vector));
        self.emit(Op::VectorLane { out, vector, lane });
        id
    }
    pub(super) fn vector_reduce_add<T: VectorElement, const LANES: u16>(
        &mut self,
        vector: VectorId<T, LANES>,
    ) -> ScalarId<T> {
        self.require_vector::<T, LANES>(crate::target::VectorOperationClass::ReduceAdd);
        let vector = self.use_vector(vector);
        let (out, id) = self.scalar_out_with::<T>(self.uniformity_of(vector));
        self.record_fact(NumericalFact::ReassociatedReduction);
        self.emit(Op::VectorReduceAdd { out, vector });
        id
    }
    pub(super) fn read<R: Representation>(
        &mut self,
        p: ReadablePlaceId<R>,
        i: &[ScalarId<Idx>],
    ) -> ScalarId<R::Element> {
        let place = self.place(p.owner(), p.kernel(), p.block(), p.index());
        let index = self.checked_index(place, i);
        let (out, id) = self.scalar_out::<R::Element>();
        self.emit(Op::Read {
            out,
            place: place.place,
            representation: place.representation,
            index,
        });
        id
    }
    pub(super) fn vector_read<R: Representation, const LANES: u16>(
        &mut self,
        p: ReadablePlaceId<R>,
        i: &[ScalarId<Idx>],
        axis: u32,
        active: ScalarId<Idx>,
    ) -> VectorId<R::Element, LANES> {
        self.require_vector::<R::Element, LANES>(crate::target::VectorOperationClass::Read {
            representation: R::id(),
        });
        let place = self.place(p.owner(), p.kernel(), p.block(), p.index());
        assert!(
            axis < place.rank,
            "vector read axis is outside the place rank"
        );
        let index = self.checked_index(place, i);
        let active = self.use_scalar(active);
        let uniformity = self.combined_uniformity(index.iter().copied().chain([active]));
        let (out, id) = self.vector_out::<R::Element, LANES>(uniformity);
        self.emit(Op::VectorRead {
            out,
            place: place.place,
            representation: place.representation,
            index,
            axis,
            active,
        });
        id
    }
    pub(super) fn vector_write<R: WritableRepresentation, const LANES: u16>(
        &mut self,
        p: WritablePlaceId<R>,
        i: &[ScalarId<Idx>],
        axis: u32,
        active: ScalarId<Idx>,
        value: VectorId<R::Element, LANES>,
    ) {
        self.require_vector::<R::Element, LANES>(crate::target::VectorOperationClass::Write {
            representation: R::id(),
        });
        let place = self.place(p.owner(), p.kernel(), p.block(), p.index());
        assert!(
            axis < place.rank,
            "vector write axis is outside the place rank"
        );
        let index = self.checked_index(place, i);
        let active = self.use_scalar(active);
        let value = self.use_vector(value);
        self.emit(Op::VectorWrite {
            place: place.place,
            representation: place.representation,
            index,
            axis,
            active,
            value,
        });
    }
    pub(super) fn write<R: WritableRepresentation>(
        &mut self,
        p: WritablePlaceId<R>,
        i: &[ScalarId<Idx>],
        v: ScalarId<R::Element>,
    ) {
        let place = self.place(p.owner(), p.kernel(), p.block(), p.index());
        let index = self.checked_index(place, i);
        let value = self.use_scalar(v);
        self.emit(Op::Write {
            place: place.place,
            representation: place.representation,
            index,
            value,
        });
    }
    pub(super) fn read_plane<R: Representation>(
        &mut self,
        p: PlaneId<R>,
        i: &[ScalarId<Idx>],
    ) -> ScalarId<U32> {
        self.assert_kernel(p.owner(), p.kernel());
        if !self.dominates(p.block(), self.block) {
            panic!("kernel builder: {p:?} is not visible in {:?}", self.block);
        }
        let (place_index, plane) = match self.state.planes.get(p.index() as usize) {
            Some(entry) => *entry,
            None => panic!("kernel builder: {p:?} is not a plane of the open kernel"),
        };
        let place = self.place(p.owner(), p.kernel(), p.block(), place_index);
        let index = self.checked_index(place, i);
        let (out, id) = self.scalar_out::<U32>();
        self.emit(Op::ReadPlane {
            out,
            place: place.place,
            plane,
            index,
        });
        id
    }
    pub(super) fn plane<R: Representation>(&mut self, p: ReadablePlaceId<R>, n: u32) -> PlaneId<R> {
        let _ = self.place(p.owner(), p.kernel(), p.block(), p.index());
        if n >= R::PLANES {
            panic!(
                "kernel builder: representation `{}` has {} planes, plane {n} requested",
                R::NAME,
                R::PLANES
            );
        }
        let index = self.state.planes.len() as u32;
        self.state.planes.push((p.index(), n));
        PlaneId::new(self.owner(), self.kernel_index(), self.block, index)
    }
    pub(super) fn extent<R: Representation>(
        &mut self,
        p: ReadablePlaceId<R>,
        a: u32,
    ) -> ScalarId<Idx> {
        let place = self.place(p.owner(), p.kernel(), p.block(), p.index());
        if a >= place.rank {
            panic!(
                "kernel builder: extent of axis {a} of a place of rank {}",
                place.rank
            );
        }
        let nat = match place.place {
            PlaceRef::Global { slot } => {
                self.assert_kernel(slot.owner(), slot.kernel());
                let view = self.state.bindings[slot.index() as usize].view;
                Some(self.views[view.index as usize].extents[a as usize])
            }
            PlaceRef::Local { index } => {
                Some(self.state.locals[index as usize].extents[a as usize])
            }
        };
        let (out, id) = self.index_out_with(nat, Uniformity::Workgroup);
        self.emit(Op::Extent {
            out,
            place: place.place,
            axis: a,
        });
        id
    }
    pub(super) fn atomic<R: WritableRepresentation>(
        &mut self,
        op: AtomicOp,
        p: WritablePlaceId<R>,
        i: &[ScalarId<Idx>],
        v: ScalarId<R::Element>,
    ) {
        let place = self.place(p.owner(), p.kernel(), p.block(), p.index());
        let index = self.checked_index(place, i);
        let value = self.use_scalar(v);
        self.emit(Op::Atomic {
            op,
            place: place.place,
            representation: place.representation,
            index,
            value,
        });
    }
    pub(super) fn store_slot<T: ScalarType>(&mut self, s: WritableScalar<T>, v: ScalarId<T>) {
        self.assert_kernel(s.owner, s.kernel);
        if s.index as usize >= self.state.result_slots.len() {
            panic!("kernel builder: {s:?} is not a result slot of the open kernel");
        }
        let raw = ErasedValue::new(v.owner(), v.kernel(), v.block(), v.index());
        assert_eq!(
            self.uniformity_of(raw),
            Uniformity::Workgroup,
            "only workgroup-uniform values may cross a kernel/schedule cut"
        );
        let value = self.use_scalar(v);
        self.emit(Op::StoreSlot {
            slot: s.index,
            value,
            election: ops::StoreElection::GlobalLeader,
        });
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
    pub(super) fn intrinsic<I: TypedIntrinsic<B>>(&mut self, args: I::Args) -> I::Result {
        let id = I::id();
        let resources = I::resources(&args, self.expr);
        if resources.requires_subgroup {
            self.state.uses_subgroup = true;
        }
        self.state.intrinsic_resources.push(resources);
        self.state.intrinsics_used.push(id);
        let result_uniformity = match registry::intrinsic_signature(id).effects.result_uniformity {
            registry::IntrinsicUniformity::Workgroup => Uniformity::Workgroup,
            registry::IntrinsicUniformity::Subgroup => Uniformity::Subgroup,
            registry::IntrinsicUniformity::Varying => Uniformity::Varying,
        };
        let mut sub = self.reborrow();
        let mut sink = IntrinsicSink {
            builder: &mut sub,
            intrinsic: id,
            result_uniformity,
        };
        I::lower(&args, &mut sink)
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

    fn define_schema(&mut self, schema: &ValueSchema) -> Vec<ErasedValue> {
        schema
            .values()
            .iter()
            .map(|ty| self.define(ty.clone()))
            .collect()
    }

    pub(super) fn branch<V: KernelValues>(
        &mut self,
        c: ScalarId<Bool>,
        t: impl FnOnce(&mut KernelBuilder<'_, B>) -> V,
        e: impl FnOnce(&mut KernelBuilder<'_, B>) -> V,
    ) -> V {
        let cond = self.use_scalar(c);
        let control_uniformity = self.state.blocks[self.block.index() as usize]
            .control_uniformity
            .combine(self.uniformity_of(cond));
        let schema = V::schema();
        let multiplicity = self.state.blocks[self.block.index() as usize].multiplicity;
        let then = self.new_block(multiplicity, control_uniformity);
        let then_values = {
            let mut sub = KernelBuilder {
                inner: self.in_block(then),
            };
            t(&mut sub).erase()
        };
        self.yield_values(then, &then_values, &schema);
        let otherwise = self.new_block(multiplicity, control_uniformity);
        let else_values = {
            let mut sub = KernelBuilder {
                inner: self.in_block(otherwise),
            };
            e(&mut sub).erase()
        };
        self.yield_values(otherwise, &else_values, &schema);
        let outs = schema
            .values()
            .iter()
            .enumerate()
            .map(|(index, ty)| {
                let uniformity = control_uniformity
                    .combine(self.uniformity_of(then_values[index]))
                    .combine(self.uniformity_of(else_values[index]));
                self.define_with(ty.clone(), None, uniformity)
            })
            .collect::<Vec<_>>();
        self.emit(Op::Branch {
            cond,
            then,
            otherwise,
            outs: outs.clone(),
        });
        V::restore(&outs)
    }

    pub(super) fn repeat<V: KernelValues>(
        &mut self,
        s: ScalarId<Idx>,
        e: ScalarId<Idx>,
        init: V,
        body: impl FnOnce(&mut KernelBuilder<'_, B>, ScalarId<Idx>, V) -> V,
    ) -> V {
        let (start, end) = (self.use_scalar(s), self.use_scalar(e));
        let range_uniformity = self.uniformity_of(start).combine(self.uniformity_of(end));
        let control_uniformity = self.state.blocks[self.block.index() as usize]
            .control_uniformity
            .combine(range_uniformity);
        let schema = V::schema();
        let init_values = init.erase();
        assert_eq!(
            init_values.len(),
            schema.len(),
            "repeat initial carry count differs from its typed schema"
        );
        let carries_in: Vec<ErasedValue> = init_values
            .into_iter()
            .zip(schema.values().iter())
            .map(|(v, ty)| self.use_value(v, ty))
            .collect();
        let trip = match (self.nat_of(start), self.nat_of(end)) {
            (Some(a), Some(b)) => {
                let m = self.expr.nat_max(b, a);
                Some(self.expr.nat_sub(m, a))
            }
            _ => None,
        };
        let multiplicity = match (
            self.state.blocks[self.block.index() as usize].multiplicity,
            trip,
        ) {
            (Some(m), Some(t)) => Some(self.expr.nat_mul(m, t)),
            _ => None,
        };
        let block = self.new_block(multiplicity, control_uniformity);
        let (binder, params, next) = {
            let mut sub = self.in_block(block);
            let binder = sub.define_with(ValueType::Index, None, range_uniformity);
            let params = carries_in
                .iter()
                .zip(schema.values())
                .map(|(initial, ty)| {
                    let uniformity = sub.uniformity_of(*initial).combine(control_uniformity);
                    sub.define_with(ty.clone(), None, uniformity)
                })
                .collect::<Vec<_>>();
            let carried = V::restore(&params);
            let mut builder = KernelBuilder { inner: sub };
            let next = body(
                &mut builder,
                ScalarId::new(binder.owner, binder.kernel, binder.block, binder.index()),
                carried,
            )
            .erase();
            (binder, params, next)
        };
        self.yield_values(block, &next, &schema);
        let outs = next
            .iter()
            .zip(schema.values())
            .map(|(value, ty)| {
                let uniformity = self.uniformity_of(*value).combine(control_uniformity);
                self.define_with(ty.clone(), None, uniformity)
            })
            .collect::<Vec<_>>();
        self.emit(Op::Repeat {
            start,
            end,
            binder,
            carries_in,
            carry_params: params,
            body: block,
            outs: outs.clone(),
        });
        V::restore(&outs)
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
        assert_eq!(
            self.state.numerical.len(),
            self.state.fact_multiplicity.len(),
            "every numerical fact must have one multiplicity"
        );
        let id = KernelId::new(owner, kernel_index);
        self.kernels.push(Kernel {
            inner: KernelData {
                owner,
                kernel: kernel_index,
                interface,
                locals: std::mem::take(&mut self.state.locals),
                blocks,
                block_multiplicity,
                value_types: std::mem::take(&mut self.state.values)
                    .into_iter()
                    .map(|v| v.ty)
                    .collect(),
                numerical: std::mem::take(&mut self.state.numerical),
                fact_multiplicity: std::mem::take(&mut self.state.fact_multiplicity),
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
fn definitely_written<B: Backend>(blocks: &[BlockData<B>], block: BlockId) -> Vec<u32> {
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

// ---------------------------------------------------------------------------
// The intrinsic sink (backend-facing surface of the builder inside a typed
// intrinsic lowering)
// ---------------------------------------------------------------------------

impl<'a, B: Backend> IntrinsicSink<'a, B> {
    pub fn addressable_resource(
        &mut self,
        class: crate::target::ResourceClassId,
        units: NatExpr,
        alignment_units: u64,
        lifetime: crate::target::ResourceLifetime,
    ) -> ops::AddressableResourceHandle {
        self.builder
            .allocate_addressable_resource(class, units, alignment_units, lifetime)
    }
    /// The kernel builder, for lowerings that expand into ordinary ops.
    pub fn kernel(&mut self) -> KernelBuilder<'_, B> {
        KernelBuilder {
            inner: self.builder.reborrow(),
        }
    }
    /// Erases a typed scalar argument for the intrinsic op.
    pub fn scalar<T: ScalarType>(&mut self, value: ScalarId<T>) -> ErasedValue {
        self.builder.use_scalar(value)
    }
    /// The place a readable argument addresses.
    pub fn readable<R: Representation>(&self, place: ReadablePlaceId<R>) -> PlaceRef {
        self.builder
            .place(place.owner(), place.kernel(), place.block(), place.index())
            .place
    }
    /// The place a writable argument addresses.
    pub fn writable<R: Representation>(&self, place: WritablePlaceId<R>) -> PlaceRef {
        self.builder
            .place(place.owner(), place.kernel(), place.block(), place.index())
            .place
    }
    /// Emits the backend intrinsic op with typed results, returning one
    /// erased value per declared result type.
    pub fn emit(
        &mut self,
        op: B::Intrinsic,
        args: Vec<ErasedValue>,
        results: &[ValueType],
    ) -> Vec<ErasedValue> {
        let signature = registry::intrinsic_signature(self.intrinsic);
        self.builder.record_intrinsic_numerics(signature, &op);
        let outs: Vec<ErasedValue> = results
            .iter()
            .map(|ty| {
                self.builder
                    .define_with(ty.clone(), None, self.result_uniformity)
            })
            .collect();
        self.builder.emit(Op::Intrinsic {
            intrinsic: self.intrinsic,
            op,
            outs: outs.clone(),
            args,
            mapping_dependencies: Vec::new(),
        });
        outs
    }
    /// Types an erased result as a scalar handle. The value must have been
    /// declared with the scalar's value type.
    pub fn result<T: ScalarType>(&mut self, value: ErasedValue) -> ScalarId<T> {
        self.builder.use_value(value, &value_type_of::<T>());
        ScalarId::new(value.owner, value.kernel, value.block, value.index())
    }
}

// Registry-validated dispatch used only by the core semantic walker for an
// authored backend lowering/helper. Result shape comes from the registry;
// the backend cannot invent it.
impl<'s, 'k, B: Backend> ops::SemanticIntrinsicSink<'s, 'k, B> {
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
    pub(crate) fn open(
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
        self.builder
            .inner
            .record_intrinsic_numerics(&self.signature, &op);
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

    pub(crate) fn finish(self) -> ops::SemanticIntrinsicResult {
        assert!(
            self.emitted,
            "intrinsic dispatcher returned without emitting"
        );
        self.result
            .expect("emitted intrinsic has one closed result")
    }
}

// ---------------------------------------------------------------------------
// KernelValues
// ---------------------------------------------------------------------------

impl super::values_sealed::Sealed for () {}
impl KernelValues for () {
    fn schema() -> ValueSchema {
        ValueSchema::new(Vec::new())
    }
    fn erase(&self) -> Vec<ErasedValue> {
        Vec::new()
    }
    fn restore(_values: &[ErasedValue]) -> Self {}
}

impl<T: ScalarType> super::values_sealed::Sealed for ScalarId<T> {}
impl<T: ScalarType> KernelValues for ScalarId<T> {
    fn schema() -> ValueSchema {
        ValueSchema::new(vec![value_type_of::<T>()])
    }
    fn erase(&self) -> Vec<ErasedValue> {
        vec![ErasedValue::new(
            self.owner(),
            self.kernel(),
            self.block(),
            self.index(),
        )]
    }
    fn restore(values: &[ErasedValue]) -> Self {
        assert_eq!(values.len(), 1, "scalar schema restores exactly one value");
        ScalarId::new(
            values[0].owner,
            values[0].kernel,
            values[0].block,
            values[0].index(),
        )
    }
}

macro_rules! tuple_values {
    ($($name:ident),+) => {
        impl<$($name: KernelValues),+> super::values_sealed::Sealed for ($($name,)+) {}
        impl<$($name: KernelValues),+> KernelValues for ($($name,)+) {
            fn schema() -> ValueSchema {
                let mut schema = Vec::new();
                $(schema.extend_from_slice($name::schema().values());)+
                ValueSchema::new(schema)
            }
            fn erase(&self) -> Vec<ErasedValue> {
                #[allow(non_snake_case)]
                let ($($name,)+) = self;
                let mut values = Vec::new();
                $(values.extend($name.erase());)+
                values
            }
            fn restore(values: &[ErasedValue]) -> Self {
                assert_eq!(values.len(), Self::schema().len(), "tuple schema restore length mismatch");
                let mut offset = 0usize;
                $(
                    #[allow(non_snake_case)]
                    let $name = {
                        let len = $name::schema().len();
                        let v = $name::restore(&values[offset..offset + len]);
                        offset += len;
                        v
                    };
                )+
                let _ = offset;
                ($($name,)+)
            }
        }
    };
}

tuple_values!(A);
tuple_values!(A, C);
tuple_values!(A, C, D);
tuple_values!(A, C, D, E);
tuple_values!(A, C, D, E, F);
tuple_values!(A, C, D, E, F, G);
tuple_values!(A, C, D, E, F, G, H);
tuple_values!(A, C, D, E, F, G, H, I);

// ---------------------------------------------------------------------------
// Closed kernels
// ---------------------------------------------------------------------------

#[derive(Debug)]
pub(crate) struct Arena<B: Backend> {
    owner: OwnerToken,
    kernels: Vec<Kernel<B>>,
}

impl<B: Backend> Arena<B> {
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

pub(crate) fn arena_from_kernels<B: Backend>(
    owner: OwnerToken,
    kernels: Vec<Kernel<B>>,
) -> KernelArena<B> {
    KernelArena {
        inner: Arena { owner, kernels },
    }
}

pub(crate) fn arena_into_kernels<B: Backend>(arena: KernelArena<B>) -> Vec<Kernel<B>> {
    arena.inner.kernels
}

pub(crate) fn data<B: Backend>(kernel: &Kernel<B>) -> &KernelData<B> {
    &kernel.inner
}

#[derive(Debug)]
pub(crate) struct KernelData<B: Backend> {
    pub(super) owner: OwnerToken,
    pub(super) kernel: u32,
    interface: KernelInterface,
    locals: Vec<LocalAllocation>,
    blocks: Vec<Block<B>>,
    block_multiplicity: Vec<Option<NatExpr>>,
    value_types: Vec<ValueType>,
    numerical: Vec<NumericalFact>,
    fact_multiplicity: Vec<Option<NatExpr>>,
    intrinsic_resources: Vec<IntrinsicResources>,
    addressable_resources: Vec<ops::AddressableResourceLease>,
    intrinsics_used: Vec<IntrinsicId>,
    resource_facts: ResourceFacts,
}

impl<B: Backend> KernelData<B> {
    fn retained_bytes(&self) -> usize {
        let interface = &self.interface;
        interface.bindings.capacity() * std::mem::size_of::<ops::Binding>()
            + interface
                .bindings
                .iter()
                .map(|binding| binding.extents.capacity() * std::mem::size_of::<NatExpr>())
                .sum::<usize>()
            + interface.nat_args.capacity() * std::mem::size_of::<NatExpr>()
            + interface.scalar_args.capacity() * std::mem::size_of::<(SymbolId, DType)>()
            + interface.result_slots.capacity()
                * std::mem::size_of::<(crate::schedule::AnyScalarSlot, DType)>()
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
            + self.value_types.capacity() * std::mem::size_of::<ValueType>()
            + self.numerical.capacity() * std::mem::size_of::<NumericalFact>()
            + self.fact_multiplicity.capacity() * std::mem::size_of::<Option<NatExpr>>()
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
    pub(super) fn value_type(&self, v: ErasedValue) -> ValueType {
        assert_eq!(
            v.owner, self.owner,
            "value belongs to another implementation"
        );
        assert_eq!(v.kernel, self.kernel, "value belongs to another kernel");
        match self.value_types.get(v.index() as usize) {
            Some(ty) => ty.clone(),
            None => panic!(
                "{v:?} is outside its kernel of {} values",
                self.value_types.len()
            ),
        }
    }
    pub(super) fn numerical_facts(&self) -> &[NumericalFact] {
        &self.numerical
    }
    pub(crate) fn resource_facts(&self) -> &ResourceFacts {
        &self.resource_facts
    }

    // ----- crate-private facts for the implementation builder ----------------

    /// Each recorded fact with the product of the trip counts of its
    /// enclosing repeats (`None` when not an arena expression).
    pub(crate) fn fact_multiplicities(
        &self,
    ) -> impl Iterator<Item = (&NumericalFact, Option<NatExpr>)> + '_ {
        self.numerical
            .iter()
            .zip(self.fact_multiplicity.iter().copied())
    }
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
    pub(crate) fn value_types(&self) -> &[ValueType] {
        &self.value_types
    }
    /// Rewrites view indices of every binding (used when a spliced child's
    /// kernels join the parent's arena).
    pub(crate) fn remap_views(
        &mut self,
        map: impl Fn(crate::storage::AnyBufferView) -> crate::storage::AnyBufferView,
    ) {
        for binding in &mut self.interface.bindings {
            binding.view = map(binding.view);
        }
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
            result.0 = map_slot(result.0);
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

fn rebrand_op<B: Backend>(
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
        | Op::ReadPlane {
            out,
            place: p,
            index,
            ..
        } => {
            *out = value(*out);
            *p = place(*p);
            values(index);
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
pub(crate) fn data_mut<B: Backend>(kernel: &mut Kernel<B>) -> &mut KernelData<B> {
    &mut kernel.inner
}
