//! The closed kernel operation vocabulary as native compilers read it
//! (spec §7). Every operation names typed handles; there is no generic
//! value enum a backend matches on to discover a category.

use super::{BindingSlot, BlockId};
use crate::identity::OwnerToken;
use crate::physical_target::PhysicalDialect;
use crate::storage::LaunchLocalKind;
use seismic_lang::expr::{NatExpr, SymbolId};
use seismic_lang::ids::{CapabilityId, IntrinsicId, RepresentationId};
use seismic_lang::intrinsics::{AtomicOp, MathOp};
use seismic_lang::types::DType;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum BinaryOp {
    Add,
    Sub,
    Mul,
    Div,
    Rem,
    Min,
    Max,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum UnaryOp {
    Neg,
    Abs,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum BitOp {
    And,
    Or,
    Xor,
    Shl,
    Shr,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum CmpOp {
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum LogicOp {
    And,
    Or,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum MathPrecision {
    Exact,
    Approximate,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum BarrierScope {
    Workgroup,
    Subgroup,
}

/// Source location of a data-dependent check.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct CheckSite {
    pub failure: seismic_lang::failure::SourceFailure,
    pub path: String,
    pub line: u32,
}

/// An erased value reference inside one kernel, valid only with the kernel
/// that owns it. Backends obtain the type through `Kernel::value_type`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ErasedValue {
    pub(crate) owner: OwnerToken,
    pub(crate) kernel: u32,
    pub(crate) block: BlockId,
    pub(crate) index: u32,
}

impl ErasedValue {
    pub(crate) fn new(owner: OwnerToken, kernel: u32, block: BlockId, index: u32) -> Self {
        Self {
            owner,
            kernel,
            block,
            index,
        }
    }
    pub fn ordinal(self) -> u32 {
        self.index
    }
    pub fn index(self) -> u32 {
        self.ordinal()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ValueType {
    Scalar(DType),
    /// A fixed-width native vector. Lanes are part of the typed IR value,
    /// never inferred by a backend emitter.
    Vector {
        dtype: DType,
        lanes: u16,
    },
    Index,
    Bool,
    /// A backend-opaque intrinsic value.
    Opaque {
        name: &'static str,
    },
}

/// A closed SSA reference paired with its construction-proved type. Native
/// emitters consume these rather than recovering categories from side maps.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ClosedValue {
    pub value: ErasedValue,
    pub ty: ValueType,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ClosedPlaceKind {
    Global {
        slot: BindingSlot,
        access: BindingAccess,
        buffer_ordinal: u32,
    },
    Local {
        index: u32,
        kind: LaunchLocalKind,
    },
}

/// One fully resolved place. Representation/rank/address-space facts are
/// attached once by core and never reconstructed by native emitters.
#[derive(Clone, Debug)]
pub struct ClosedPlace<G = crate::physical_target::RepresentationGeometry> {
    pub raw: PlaceRef,
    pub kind: ClosedPlaceKind,
    pub representation: RepresentationId,
    pub rank: u32,
    pub extents: Vec<NatExpr>,
    pub geometry: G,
    pub words: ClosedPlaceWords,
    pub realization: Option<crate::physical_target::LocalRealization>,
}

impl<G> ClosedPlace<G> {
    pub(crate) fn map_geometry<H>(self, geometry: H) -> ClosedPlace<H> {
        ClosedPlace {
            raw: self.raw,
            kind: self.kind,
            representation: self.representation,
            rank: self.rank,
            extents: self.extents,
            geometry,
            words: self.words,
            realization: self.realization,
        }
    }
}

/// A global binding place proved at construction.  Operations whose ABI is
/// intrinsically global (such as one-shot representation conversion) use
/// this projection so native emitters never branch on a local alternative.
#[derive(Clone, Debug)]
pub struct ClosedGlobalPlace<G = crate::physical_target::RepresentationGeometry> {
    pub slot: BindingSlot,
    pub access: BindingAccess,
    pub buffer_ordinal: u32,
    pub representation: RepresentationId,
    pub rank: u32,
    pub extents: Vec<NatExpr>,
    pub geometry: G,
    pub words: crate::physical_target::BindingWordLayout,
}

impl<G> ClosedGlobalPlace<G> {
    pub(crate) fn map_geometry<H>(self, geometry: H) -> ClosedGlobalPlace<H> {
        ClosedGlobalPlace {
            slot: self.slot,
            access: self.access,
            buffer_ordinal: self.buffer_ordinal,
            representation: self.representation,
            rank: self.rank,
            extents: self.extents,
            geometry,
            words: self.words,
        }
    }
}

pub type ClosedReadablePlace = ClosedPlace<crate::physical_target::ReadableRepresentationGeometry>;
pub type ClosedDensePlace = ClosedPlace<crate::physical_target::DenseRepresentationGeometry>;
pub type ClosedPackedPlace = ClosedPlace<crate::physical_target::PackedRepresentationGeometry>;
pub type ClosedExternalGlobalPlace =
    ClosedGlobalPlace<crate::physical_target::ExternalRepresentationGeometry>;
pub type ClosedPackedGlobalPlace =
    ClosedGlobalPlace<crate::physical_target::PackedRepresentationGeometry>;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ClosedPlaceWords {
    Binding(crate::physical_target::BindingWordLayout),
    Local(crate::physical_target::LocalWordLayout),
}

/// An index value whose category was proved when the kernel was closed.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ClosedIndexValue(pub(crate) ClosedValue);

impl ClosedIndexValue {
    pub fn value(self) -> ErasedValue {
        self.0.value
    }
}

/// A boolean value whose category was proved when the kernel was closed.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ClosedBoolValue(pub(crate) ClosedValue);

impl ClosedBoolValue {
    pub fn value(self) -> ErasedValue {
        self.0.value
    }
}

/// Core-resolved, operation-shaped view of the closed kernel IR. Each variant
/// carries the exact typed operands and resolved places it needs; native
/// emitters never recover categories, ranks, representations, or intrinsic
/// signatures from side tables.
pub enum ClosedOpView<'a, B: PhysicalDialect> {
    Constant {
        out: ClosedValue,
        value: ConstantValue,
    },
    Binary {
        op: BinaryOp,
        out: ClosedValue,
        a: ClosedValue,
        b: ClosedValue,
    },
    Unary {
        op: UnaryOp,
        out: ClosedValue,
        a: ClosedValue,
    },
    Bit {
        op: BitOp,
        out: ClosedValue,
        a: ClosedValue,
        b: ClosedValue,
    },
    Fma {
        out: ClosedValue,
        a: ClosedValue,
        b: ClosedValue,
        c: ClosedValue,
    },
    VectorSplat {
        out: ClosedValue,
        value: ClosedValue,
    },
    /// Ordered, bit-preserving assembly of scalar lanes.
    VectorFromLanes {
        out: ClosedValue,
        lanes: Vec<ClosedValue>,
    },
    VectorBinary {
        op: BinaryOp,
        out: ClosedValue,
        a: ClosedValue,
        b: ClosedValue,
    },
    VectorUnary {
        op: UnaryOp,
        out: ClosedValue,
        a: ClosedValue,
    },
    VectorBit {
        op: BitOp,
        out: ClosedValue,
        a: ClosedValue,
        b: ClosedValue,
    },
    VectorFma {
        out: ClosedValue,
        a: ClosedValue,
        b: ClosedValue,
        c: ClosedValue,
    },
    VectorCast {
        out: ClosedValue,
        a: ClosedValue,
        to: ValueType,
    },
    VectorLane {
        out: ClosedValue,
        vector: ClosedValue,
        lane: u16,
    },
    VectorReduceAdd {
        out: ClosedValue,
        vector: ClosedValue,
    },
    ApproximateMath {
        op: MathOp,
        out: ClosedValue,
        a: ClosedValue,
    },
    Cast {
        out: ClosedValue,
        a: ClosedValue,
        to: ValueType,
    },
    Bitcast {
        out: ClosedValue,
        a: ClosedValue,
        to: ValueType,
    },
    /// Narrow scalar payload, zero-extended to U32 without numerical conversion.
    ScalarBits {
        out: ClosedValue,
        a: ClosedValue,
    },
    /// The low 16 payload bits interpreted as the declared narrow scalar type.
    ScalarFromBits {
        out: ClosedValue,
        a: ClosedValue,
    },
    Cmp {
        op: CmpOp,
        out: ClosedBoolValue,
        a: ClosedValue,
        b: ClosedValue,
    },
    Select {
        out: ClosedValue,
        condition: ClosedBoolValue,
        a: ClosedValue,
        b: ClosedValue,
    },
    Logic {
        op: LogicOp,
        out: ClosedBoolValue,
        a: ClosedBoolValue,
        b: ClosedBoolValue,
    },
    Not {
        out: ClosedBoolValue,
        a: ClosedBoolValue,
    },
    Geometry {
        out: ClosedIndexValue,
        kind: GeometryValue,
    },
    NatArg {
        out: ClosedIndexValue,
        index: u32,
        expression: NatExpr,
    },
    ScalarArg {
        out: ClosedValue,
        index: u32,
        symbol: SymbolId,
        kind: crate::repr::ScalarKind,
    },
    Read {
        out: ClosedValue,
        place: ClosedDensePlace,
        indices: Vec<ClosedIndexValue>,
    },
    /// Reads consecutive logical elements on one axis. Lanes greater than or
    /// equal to `active` are exactly zero. Packed vector reads are expanded
    /// before closure into guarded typed field reads and scalar recipes.
    VectorRead {
        out: ClosedValue,
        place: ClosedDensePlace,
        indices: Vec<ClosedIndexValue>,
        axis: u32,
        active: ClosedIndexValue,
    },
    /// Writes consecutive logical elements on one axis. Only lanes whose
    /// ordinal is less than `active` are written.
    VectorWrite {
        place: ClosedDensePlace,
        indices: Vec<ClosedIndexValue>,
        axis: u32,
        active: ClosedIndexValue,
        value: ClosedValue,
    },
    /// A typed field of the packet containing the indexed logical element.
    /// Code fields yield zero-extended U32; dense fields preserve their dtype.
    ReadPlaneField {
        out: ClosedValue,
        place: ClosedPackedPlace,
        plane: u32,
        field: u32,
        plane_info: seismic_lang::registry::PlaneInfo,
        indices: Vec<ClosedIndexValue>,
    },
    ReadPlane {
        out: ClosedValue,
        place: ClosedPackedPlace,
        plane: u32,
        plane_info: seismic_lang::registry::PlaneInfo,
        indices: Vec<ClosedIndexValue>,
        /// Storage element within the named plane of the addressed packet.
        element: ClosedIndexValue,
    },
    RepresentationConvertPacket {
        source: ClosedExternalGlobalPlace,
        destination: ClosedPackedGlobalPlace,
        conversion: seismic_lang::ids::RepresentationConversionId,
        recipe: &'static seismic_lang::registry::RepresentationConversion,
        packet: ClosedIndexValue,
    },
    Write {
        place: ClosedDensePlace,
        indices: Vec<ClosedIndexValue>,
        value: ClosedValue,
    },
    Extent {
        out: ClosedIndexValue,
        place: ClosedPlace,
        axis: u32,
    },
    Atomic {
        op: AtomicOp,
        place: ClosedDensePlace,
        indices: Vec<ClosedIndexValue>,
        value: ClosedValue,
    },
    StoreSlot {
        slot: u32,
        kind: crate::repr::ScalarKind,
        value: ClosedValue,
        election: StoreElection,
    },
    Barrier(BarrierScope),
    Intrinsic {
        intrinsic: IntrinsicId,
        signature: &'static seismic_lang::registry::IntrinsicSignature,
        op: &'a B::Intrinsic,
        outputs: Vec<ClosedValue>,
        arguments: Vec<ClosedValue>,
        mapping_dependencies: Vec<ClosedIndexValue>,
    },
    Branch {
        condition: ClosedBoolValue,
        then_block: BlockId,
        else_block: BlockId,
        outputs: Vec<ClosedValue>,
        then_yields: Vec<ClosedValue>,
        else_yields: Vec<ClosedValue>,
    },
    Repeat {
        start: ClosedIndexValue,
        end: ClosedIndexValue,
        binder: ClosedIndexValue,
        carries_in: Vec<ClosedValue>,
        carry_parameters: Vec<ClosedValue>,
        body: BlockId,
        outputs: Vec<ClosedValue>,
        body_yields: Vec<ClosedValue>,
    },
    Yield {
        values: Vec<ClosedValue>,
    },
}

/// Schema of a join/carry tuple.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct ValueSchema(Vec<ValueType>);

impl ValueSchema {
    pub(crate) fn new(values: Vec<ValueType>) -> Self {
        Self(values)
    }
    pub(crate) fn values(&self) -> &[ValueType] {
        &self.0
    }
    pub(crate) fn len(&self) -> usize {
        self.0.len()
    }
}

/// A place reference: what a read/write addresses.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum PlaceRef {
    /// A global view bound at `slot`.
    Global { slot: BindingSlot },
    /// A launch-local allocation of this kernel.
    Local { index: u32 },
}

/// The kernel's argument table.
#[derive(Clone, Debug)]
pub struct KernelInterface {
    pub bindings: Vec<Binding>,
    /// Scalar arguments bound from arena expressions, in slot order.
    pub nat_args: Vec<NatExpr>,
    /// Scalar arguments bound from call scalar symbols.
    pub scalar_args: Vec<(SymbolId, crate::repr::ScalarKind)>,
    /// Result slots the kernel writes back.
    pub result_slots: Vec<crate::schedule::AnyScalarSlot>,
    pub uses_subgroup: bool,
}

#[derive(Clone, Debug)]
pub struct Binding {
    pub slot: BindingSlot,
    pub view: crate::storage::AnyBufferView,
    pub access: BindingAccess,
    pub rank: u32,
    pub extents: Vec<NatExpr>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum BindingAccess {
    Read,
    Write,
}

/// One structured block: a straight-line op sequence with nested control.
#[derive(Debug)]
pub struct Block<B: PhysicalDialect> {
    pub ops: Vec<Op<B>>,
}

/// The closed operation set.
#[derive(Debug)]
pub enum Op<B: PhysicalDialect> {
    Constant {
        out: ErasedValue,
        value: ConstantValue,
    },
    Binary {
        op: BinaryOp,
        out: ErasedValue,
        a: ErasedValue,
        b: ErasedValue,
    },
    Unary {
        op: UnaryOp,
        out: ErasedValue,
        a: ErasedValue,
    },
    Bit {
        op: BitOp,
        out: ErasedValue,
        a: ErasedValue,
        b: ErasedValue,
    },
    Fma {
        out: ErasedValue,
        a: ErasedValue,
        b: ErasedValue,
        c: ErasedValue,
    },
    VectorSplat {
        out: ErasedValue,
        value: ErasedValue,
    },
    VectorFromLanes {
        out: ErasedValue,
        lanes: Vec<ErasedValue>,
    },
    VectorBinary {
        op: BinaryOp,
        out: ErasedValue,
        a: ErasedValue,
        b: ErasedValue,
    },
    VectorUnary {
        op: UnaryOp,
        out: ErasedValue,
        a: ErasedValue,
    },
    VectorBit {
        op: BitOp,
        out: ErasedValue,
        a: ErasedValue,
        b: ErasedValue,
    },
    VectorFma {
        out: ErasedValue,
        a: ErasedValue,
        b: ErasedValue,
        c: ErasedValue,
    },
    VectorCast {
        out: ErasedValue,
        a: ErasedValue,
        to: ValueType,
    },
    VectorLane {
        out: ErasedValue,
        vector: ErasedValue,
        lane: u16,
    },
    VectorReduceAdd {
        out: ErasedValue,
        vector: ErasedValue,
    },
    Math {
        op: MathOp,
        precision: MathPrecision,
        out: ErasedValue,
        a: ErasedValue,
    },
    Cast {
        out: ErasedValue,
        a: ErasedValue,
        to: ValueType,
    },
    /// Same-width bit reinterpretation. Unlike `Cast`, this performs no
    /// numerical conversion and is used by registry-versioned reference
    /// math recipes.
    Bitcast {
        out: ErasedValue,
        a: ErasedValue,
        to: ValueType,
    },
    /// Raw narrow floating payload transport. 32-bit cases normalize to Bitcast;
    /// Boolean cases normalize to Select/Cmp in the ordinary constructor.
    ScalarBits {
        out: ErasedValue,
        a: ErasedValue,
    },
    ScalarFromBits {
        out: ErasedValue,
        a: ErasedValue,
    },
    Cmp {
        op: CmpOp,
        out: ErasedValue,
        a: ErasedValue,
        b: ErasedValue,
    },
    Select {
        out: ErasedValue,
        cond: ErasedValue,
        a: ErasedValue,
        b: ErasedValue,
    },
    Logic {
        op: LogicOp,
        out: ErasedValue,
        a: ErasedValue,
        b: ErasedValue,
    },
    Not {
        out: ErasedValue,
        a: ErasedValue,
    },
    Geometry {
        out: ErasedValue,
        kind: GeometryValue,
    },
    NatArg {
        out: ErasedValue,
        index: u32,
    },
    ScalarArg {
        out: ErasedValue,
        index: u32,
    },
    Read {
        out: ErasedValue,
        place: PlaceRef,
        representation: RepresentationId,
        index: Vec<ErasedValue>,
    },
    VectorRead {
        out: ErasedValue,
        place: PlaceRef,
        representation: RepresentationId,
        index: Vec<ErasedValue>,
        axis: u32,
        active: ErasedValue,
    },
    VectorWrite {
        place: PlaceRef,
        representation: RepresentationId,
        index: Vec<ErasedValue>,
        axis: u32,
        active: ErasedValue,
        value: ErasedValue,
    },
    ReadPlaneField {
        out: ErasedValue,
        place: PlaceRef,
        plane: u32,
        field: u32,
        index: Vec<ErasedValue>,
    },
    ReadPlane {
        out: ErasedValue,
        place: PlaceRef,
        plane: u32,
        index: Vec<ErasedValue>,
        element: ErasedValue,
    },
    /// Sealed one-shot external-packet to resident-packet initialization.
    /// This is the only operation allowed to write a read-only packed
    /// representation and is constructed exclusively from a registered exact
    /// conversion recipe.
    RepresentationConvertPacket {
        source: BindingSlot,
        destination: BindingSlot,
        conversion: seismic_lang::ids::RepresentationConversionId,
        packet: ErasedValue,
    },
    Write {
        place: PlaceRef,
        representation: RepresentationId,
        index: Vec<ErasedValue>,
        value: ErasedValue,
    },
    Extent {
        out: ErasedValue,
        place: PlaceRef,
        axis: u32,
    },
    Atomic {
        op: AtomicOp,
        place: PlaceRef,
        representation: RepresentationId,
        index: Vec<ErasedValue>,
        value: ErasedValue,
    },
    StoreSlot {
        slot: u32,
        value: ErasedValue,
        election: StoreElection,
    },
    Barrier(BarrierScope),
    Intrinsic {
        intrinsic: IntrinsicId,
        op: B::Intrinsic,
        outs: Vec<ErasedValue>,
        args: Vec<ErasedValue>,
        mapping_dependencies: Vec<ErasedValue>,
    },
    Branch {
        cond: ErasedValue,
        then: BlockId,
        otherwise: BlockId,
        outs: Vec<ErasedValue>,
    },
    Repeat {
        start: ErasedValue,
        end: ErasedValue,
        binder: ErasedValue,
        carries_in: Vec<ErasedValue>,
        carry_params: Vec<ErasedValue>,
        body: BlockId,
        outs: Vec<ErasedValue>,
    },
    /// Block terminator yielding the join/carry values to the parent.
    Yield {
        values: Vec<ErasedValue>,
    },
}

impl<B: PhysicalDialect> Op<B> {
    /// Values defined in the containing block by this actual operation.
    /// Nested repeat parameters belong to the body and are not parent results.
    pub fn defined_values(&self) -> &[ErasedValue] {
        match self {
            Self::Constant { out, .. }
            | Self::Binary { out, .. }
            | Self::Unary { out, .. }
            | Self::Bit { out, .. }
            | Self::Fma { out, .. }
            | Self::VectorSplat { out, .. }
            | Self::VectorFromLanes { out, .. }
            | Self::VectorBinary { out, .. }
            | Self::VectorUnary { out, .. }
            | Self::VectorBit { out, .. }
            | Self::VectorFma { out, .. }
            | Self::VectorCast { out, .. }
            | Self::VectorLane { out, .. }
            | Self::VectorReduceAdd { out, .. }
            | Self::Math { out, .. }
            | Self::Cast { out, .. }
            | Self::Bitcast { out, .. }
            | Self::ScalarBits { out, .. }
            | Self::ScalarFromBits { out, .. }
            | Self::Cmp { out, .. }
            | Self::Select { out, .. }
            | Self::Logic { out, .. }
            | Self::Not { out, .. }
            | Self::Geometry { out, .. }
            | Self::NatArg { out, .. }
            | Self::ScalarArg { out, .. }
            | Self::Read { out, .. }
            | Self::VectorRead { out, .. }
            | Self::ReadPlaneField { out, .. }
            | Self::ReadPlane { out, .. }
            | Self::Extent { out, .. } => std::slice::from_ref(out),
            Self::Intrinsic { outs, .. }
            | Self::Branch { outs, .. }
            | Self::Repeat { outs, .. } => outs,
            Self::VectorWrite { .. }
            | Self::RepresentationConvertPacket { .. }
            | Self::Write { .. }
            | Self::Atomic { .. }
            | Self::StoreSlot { .. }
            | Self::Yield { .. }
            | Self::Barrier(_) => &[],
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum StoreElection {
    GlobalLeader,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ConstantValue {
    F32(f32),
    /// Exact IEEE binary16 payload.
    F16(u16),
    /// Exact bfloat16 payload.
    BF16(u16),
    I32(i32),
    U32(u32),
    Bool(bool),
    Index(u64),
}

impl ConstantValue {
    /// Transport a language scalar payload into native IR without numeric conversion.
    pub fn from_scalar(value: seismic_lang::reference_math::ReferenceScalar) -> Self {
        use seismic_lang::reference_math::ReferenceScalar;
        match value {
            ReferenceScalar::F32(bits) => Self::F32(f32::from_bits(bits)),
            ReferenceScalar::F16(bits) => Self::F16(bits),
            ReferenceScalar::BF16(bits) => Self::BF16(bits),
            ReferenceScalar::I32(value) => Self::I32(value),
            ReferenceScalar::U32(value) => Self::U32(value),
            ReferenceScalar::Bool(value) => Self::Bool(value),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum GeometryValue {
    WorkgroupId(u8),
    LocalId(u8),
    GlobalId(u8),
    WorkgroupSize(u8),
    GridSize(u8),
    SubgroupLane,
    /// Stable ordinal of the actual subgroup inside the workgroup.
    SubgroupOrdinal,
    /// Actual native subgroup width of the compiled kernel.
    SubgroupSize,
}

/// Resource facts of one kernel derived at close.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct ResourceFacts {
    pub local_kinds: Vec<LaunchLocalKind>,
    pub barriers: u32,
    pub uses_subgroup: bool,
    pub binding_count: u32,
}

/// Launch-local resources one intrinsic lowering requires.
#[derive(Clone, Debug, Default)]
pub struct IntrinsicResources {
    /// Non-addressable reservations charged to the canonical launch totals.
    /// Any intrinsic that needs addressable scratch must declare an ordinary
    /// typed kernel local and receive its `PlaceRef`; these fields must never
    /// be reinterpreted as hidden offsets by an emitter.
    pub workgroup_bytes: Option<NatExpr>,
    pub participant_bytes: Option<NatExpr>,
    pub register_bytes: Option<NatExpr>,
    pub requires_subgroup: bool,
}

/// Owner- and kernel-branded lease of native addressable intrinsic state.
/// Backends may retain this opaque handle in their intrinsic enum; only core
/// resolves it to the canonical class/offset/extent projection.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct AddressableResourceHandle {
    pub(crate) owner: crate::identity::OwnerToken,
    pub(crate) kernel: u32,
    pub(crate) lease: u32,
}

impl AddressableResourceHandle {
    pub fn ordinal(self) -> u32 {
        self.lease
    }
}

#[derive(Clone, Debug)]
pub struct AddressableResourceLease {
    pub handle: AddressableResourceHandle,
    pub class_id: crate::physical_target::ResourceClassId,
    pub class: crate::physical_target::AddressableResourceClass,
    pub offset_units: NatExpr,
    pub units: NatExpr,
    pub alignment_units: u64,
    pub lifetime: crate::physical_target::ResourceLifetime,
}

/// Native-emission projection of one lease. Symbolic offset/extent values
/// are supplied through the canonical word table; the emitter receives the
/// exact class, ownership, alignment, and lifetime proved at construction.
#[derive(Clone, Debug)]
pub struct ClosedAddressableResource {
    pub handle: AddressableResourceHandle,
    pub class_id: crate::physical_target::ResourceClassId,
    pub class: crate::physical_target::AddressableResourceClass,
    pub offset_word: u32,
    pub units_word: u32,
    pub alignment_units: u64,
    pub lifetime: crate::physical_target::ResourceLifetime,
}

/// Registry-typed operands passed by the one core semantic walker to a
/// backend's authored intrinsic dispatcher. Handles are opaque outside the
/// compiler; the sink validates every projection against the canonical
/// signature before an intrinsic op can be emitted.
#[derive(Clone, Copy, Debug)]
pub struct SemanticScalar {
    pub(crate) value: super::internals::PortableValue,
    pub(crate) dtype: DType,
    pub(crate) index: bool,
    /// Strongest execution scope at which the value is equal. Subgroup and
    /// varying values are kernel-local and cannot be published through a
    /// schedule slot.
    pub(crate) uniformity: seismic_lang::registry::IntrinsicUniformity,
}
impl SemanticScalar {
    pub fn dtype(&self) -> DType {
        self.dtype
    }
    pub fn index(&self) -> bool {
        self.index
    }
    pub fn uniformity(&self) -> seismic_lang::registry::IntrinsicUniformity {
        self.uniformity
    }
}

#[derive(Clone, Copy, Debug)]
pub struct SemanticOpaque {
    pub(crate) value: super::internals::PortableValue,
    pub(crate) capability: CapabilityId,
    pub(crate) name: &'static str,
    pub(crate) uniformity: seismic_lang::registry::IntrinsicUniformity,
}
impl SemanticOpaque {
    pub fn value(&self) -> super::internals::PortableValue {
        self.value
    }
    pub fn capability(&self) -> CapabilityId {
        self.capability
    }
    pub fn name(&self) -> &'static str {
        self.name
    }
    pub fn uniformity(&self) -> seismic_lang::registry::IntrinsicUniformity {
        self.uniformity
    }
}

#[derive(Clone, Debug)]
pub struct SemanticPlace {
    pub(crate) tensor: super::internals::PortableTensor,
    pub(crate) representation: RepresentationId,
    pub(crate) rank: u32,
    pub(crate) writable: bool,
}
impl SemanticPlace {
    pub fn representation(&self) -> RepresentationId {
        self.representation
    }
    pub fn rank(&self) -> u32 {
        self.rank
    }
    pub fn writable(&self) -> bool {
        self.writable
    }
}

/// Canonical logical-index map carried by a semantic tensor intrinsic
/// operand. PhysicalDialect intrinsic variants may retain this closed map and apply
/// it to their logical element/tile coordinates; they never recover strides
/// or view transforms from schedule/global topology.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LogicalTensorMap {
    pub base: PlaceRef,
    pub representation: RepresentationId,
    /// Actual logical extents, including values produced on the device.
    /// Analytical host expressions, when available, belong to these SSA values.
    pub extents: Vec<ErasedValue>,
    pub steps: Vec<LogicalViewStep>,
}

impl LogicalTensorMap {
    /// Kernel SSA values that participate in this logical address map.  The
    /// intrinsic op carries these as ordinary operands so dominance,
    /// importing, canonical identity, and numerical traversal see the exact
    /// address computation rather than a hidden side graph.
    pub(crate) fn dependencies(&self) -> Vec<ErasedValue> {
        let mut values = Vec::new();
        let mut push = |value: ErasedValue| {
            if !values.contains(&value) {
                values.push(value);
            }
        };
        for value in &self.extents {
            push(*value);
        }
        for step in &self.steps {
            match step {
                LogicalViewStep::Slice(axes) => {
                    for axis in axes {
                        match axis {
                            LogicalSliceAxis::Point(value) => push(*value),
                            LogicalSliceAxis::Range { start, end } => {
                                push(*start);
                                push(*end);
                            }
                            LogicalSliceAxis::Full => {}
                        }
                    }
                }
                LogicalViewStep::Transpose(_) | LogicalViewStep::Plane { .. } => {}
                LogicalViewStep::Reshape { from, to } => {
                    for value in from.iter().chain(to) {
                        push(*value);
                    }
                }
            }
        }
        values
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum LogicalViewStep {
    /// Select physical storage elements from a packed representation plane.
    /// The axis belongs to this exact point in the ordered view composition.
    Plane {
        plane: u32,
        axis: u32,
    },
    Slice(Vec<LogicalSliceAxis>),
    Transpose(Vec<u32>),
    Reshape {
        from: Vec<ErasedValue>,
        to: Vec<ErasedValue>,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LogicalSliceAxis {
    Point(ErasedValue),
    Range {
        start: ErasedValue,
        end: ErasedValue,
    },
    Full,
}

#[derive(Clone, Debug)]
pub enum SemanticIntrinsicOperand {
    Scalar(SemanticScalar),
    Readable(SemanticPlace),
    Writable(SemanticPlace),
    Constant(SemanticScalar),
    Opaque(SemanticOpaque),
}

pub struct SemanticIntrinsicCall<'a> {
    pub signature: &'a seismic_lang::registry::IntrinsicSignature,
    pub operands: &'a [SemanticIntrinsicOperand],
    /// Caller-owned destination for an `Owned` result. Other result kinds
    /// must leave this empty.
    pub destination: Option<SemanticPlace>,
}

/// Logical launch domain established by an enclosing semantic parallel loop.
/// Intrinsics consume this domain; they never invent a standalone grid from
/// their operands (lane-only intrinsics may have no tensor operand at all).
#[derive(Clone, Copy, Debug)]
pub struct SegmentLaunchDomain {
    pub mode: crate::schedule::LaunchParticipation,
    pub grid: [NatExpr; 3],
    pub workgroup: [NatExpr; 3],
    pub empty: seismic_lang::expr::BoolExpr,
    /// Number of logical parallel iterations. Participation requirements from
    /// the intrinsic registry constrain this against the subgroup/workgroup.
    pub parallel_extent: NatExpr,
}

/// Additional launch requirements contributed by one authored intrinsic.
/// Core intersects these with the enclosing segment domain and the registry's
/// exact participation requirement before the segment can close.
#[derive(Clone, Copy, Debug, Default)]
pub struct SemanticIntrinsicLaunchRequirements {
    pub required_mode: Option<crate::schedule::LaunchParticipation>,
    pub required_workgroup: Option<[NatExpr; 3]>,
}

pub struct SemanticIntrinsicSink<'s, 'k, B: PhysicalDialect> {
    pub(crate) builder: &'s mut super::internals::PortableBuilder<'k, B>,
    pub(crate) signature: seismic_lang::registry::IntrinsicSignature,
    pub(crate) emitted: bool,
    pub(crate) result: Option<SemanticIntrinsicResult>,
    pub(crate) semantic_arguments: Vec<ErasedValue>,
    pub(crate) mapping_dependencies: Vec<ErasedValue>,
}

#[derive(Clone, Debug)]
pub enum SemanticIntrinsicResult {
    Void,
    Scalar(SemanticScalar),
    Owned(SemanticPlace),
    Opaque(SemanticOpaque),
}

impl SemanticScalar {
    pub fn value(&self) -> super::internals::PortableValue {
        self.value
    }
}
