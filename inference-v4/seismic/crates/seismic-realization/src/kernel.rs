//! Closed typed kernel blocks and the universal kernel operation algebra
//! (package K1 owns formation; C0 froze the dialect and intrinsic contracts
//! that backend lanes B1 code against).
//!
//! Universal semantic lowering is core-owned and shared by all backends:
//! `KernelOp<I>` is `Core(CoreKernelOp)` or `Intrinsic(I)`. Backend dialects
//! contribute only typed capability/architecture intrinsics. Every operand
//! and result carries its type; no backend may reconstruct a missing dtype,
//! shape, axis, representation, coordinate, or operand.
//!
//! # Semantics an encoder implements (exact)
//!
//! Values. A `KernelValueRef` is a by-value kernel scalar: an external input
//! (`Input`, a scalar-routed interface input), a block-local SSA value
//! (`Ssa`, defined by exactly one op before every use), or an iteration axis
//! coordinate (`Axis`, the block's `LinearIterationMap` coordinate of that
//! axis, an `i32`). Every SSA has one `KernelValueType`: a scalar dtype or a
//! backend-owned capability value.
//!
//! Places. A `KernelPlaceRef` is an addressable tensor of the closed
//! interface (`Input`, `Output`, `Local`), always in its *view coordinates*:
//! the logical coordinates of the routed value (rank = the value's tensor
//! rank). The route's `ViewTransformTemplate` (residence coordinates) is
//! applied by the encoder mechanically; kernel formation never linearizes
//! addresses. In-block view transforms (transpose, reshape, slice nodes that
//! do not cross a cut) are lowered here to coordinate arithmetic over the
//! source place, so an encoder sees only places and coordinates.
//!
//! Scalar ops. `Unary`/`Binary`/`Compare`/`Math`/`Fma`/`Cast`/`Select`
//! evaluate at `dtype` with operands converted exactly to `dtype` (narrow
//! floats widen exactly; comparisons and logic evaluate at the operand
//! dtype and produce `bool`). Integer add/sub/mul wrap at 32 bits; `Div`/
//! `Rem` are Euclidean; `Shl`/`Shr` take counts in `0..32`, `Shr` is
//! arithmetic on `i32` and logical on `u32`; float arithmetic rounds once at
//! `dtype`; `Math` is the versioned registry software sequence; `Fma` rounds
//! once. `TableLookup` indexes a constant `i32` table with an in-range
//! `u32` code (the packed-code interpretation of the registry).
//!
//! Tensor access. `Load`/`Store` read/write one dense element. `Atomic` is a
//! read-modify-write of one element with the registry `AtomicOp` law under
//! `mode`: `Serialized` (one participant owns the domain: plain
//! load/combine/round/store) or `Device` (native fetch-add/max/min for 32-bit
//! integers, native f32 add where the target has it, otherwise a
//! compare-exchange loop on the bits). Every visit's update is applied;
//! `add` rounds once at the element dtype per update; `max`/`min` are exact
//! and ignore a NaN operand.
//! `PackedPlaneRead` reads one entry of one representation plane for the
//! logical element at `coords`: entry `planes[plane].entry(v, entry)` of the
//! packing-axis coordinate `v`, zero-extended to the plane's storage dtype.
//! `PlaneLoad`/`PlaneStore` address one storage element of one plane by the
//! outer (non-packing) coordinates plus the storage-element ordinal along the
//! packing axis (bulk representation copies). No op decodes: a decode is
//! expanded here into plane reads and scalar ops.
//!
//! Control. `Repeat` is an ascending serial loop over `[start, end)` binding
//! `binder` (`i32`) in its body; SSAs defined in a body are not visible
//! after it, except through `Carry`: a `Carry` listed before the body ops
//! rebinds `current` on every visit (from `initial` on the first, from
//! `update` after each visit) and defines `result` after the loop as the
//! final value. `Branch` executes one body and defines every `joined` from
//! the taken side. `Fold` is the registry serial reduction of one axis of a
//! place at the fixed outer coordinates: ascending visits, the identity,
//! accumulator, result dtype, and tie rule of `schema` (`reduce_schema`).
//!
//! Safety. `Check` evaluates `predicate`; when it holds the `guarded` ops
//! execute; when it fails the block records the obligation in `status`
//! (first error wins) and skips `guarded`, whose SSAs then hold an
//! unspecified value of their type (the invocation fails after completion;
//! every op that can trap or address memory is itself guarded, so no skipped
//! value reaches an unguarded trap). Several `Check` sites may name one
//! status field when a guarded view is accessed at several sites.
//!
//! Publication. `Publish` writes a kernel scalar to a scalar-routed output.
//! A tensor-routed output is written by `Store`/`PlaneStore` sites.
//!
//! Participants and traversal. The block's `LinearIterationMap` says how
//! participants cover the block axes. `Traversal::DynamicPull` needs no op:
//! the encoder emits `while ((lin = atomic_fetch_add(counter, 1)) < total)`
//! over the linear coordinate, where `counter` is the launch's pull-counter
//! storage (P1 binds it from `RoutedBlock.pull_counter`) and the tail mask
//! is false. A `ParticipantPolicy::GridCooperative` block is encoded as a
//! cooperative launch; `GridBarrier` is grid-wide synchronization
//! (`grid.sync()`), emitted only in such blocks between a participant's
//! `Store` into a kernel-local intermediate (`KernelPlaceRef::Local`, a
//! device-arena kernel-local residence D1 declares for an intra-launch
//! whole-result edge) and the first `Load` of that intermediate by the
//! consuming domain. `Barrier` is the workgroup/subgroup form.

use crate::failure::{CompilerDefect, Package};
use crate::ids::{
    KernelAxisId, KernelInputId, KernelLocalId, KernelOutputId, KernelSsaId, ObligationRef,
    StatusFieldTemplateId,
};
use crate::residence::{ClosedKernelInterface, StoragePlane};
use seismic_lang::intrinsics::{AtomicOp, IntrinsicId, MathOp, PlaneField, ReduceOp, ReduceSchema};
use seismic_lang::logical::IdVec;
use seismic_lang::repr;
use seismic_lang::sir::IntrinsicUse;
use seismic_lang::syntax::ast::{BinaryOp, UnaryOp};
use seismic_lang::types::{
    CapabilityValueType, DType, Elem, ExtentExpr, NonEmpty, RuntimeExtentId, TensorType,
};

/// A kernel value reference: external input, block-local SSA, or axis.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum KernelValueRef {
    Input(KernelInputId),
    Ssa(KernelSsaId),
    Axis(KernelAxisId),
}

/// An addressable place of a kernel block: a tensor input, output, or local.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum KernelPlaceRef {
    Input(KernelInputId),
    Output(KernelOutputId),
    Local(KernelLocalId),
}

/// The type of one kernel SSA value: a scalar dtype, or a backend-owned
/// capability value (a matrix fragment) that only intrinsics produce and
/// consume.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum KernelValueType {
    Scalar(DType),
    Capability(CapabilityValueType),
}

/// A bit-exact constant value. Floats are carried as their `f64` bits so the
/// op algebra is `Eq`/`Hash`; the typed dtype states the rounding target.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ConstantValue {
    Int(i64),
    /// `f64::to_bits` of the literal; converted by value to `dtype`.
    Float { bits: u64 },
    Bool(bool),
}

/// A typed constant.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct TypedConstant {
    pub value: ConstantValue,
    pub dtype: DType,
}

/// A relational comparison.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum RelOp {
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
}

/// A retained safety predicate lowered inside a block. Every value is an
/// `i32` kernel scalar unless the variant states a dtype.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CheckPredicate {
    /// `0 <= index < extent`.
    IndexInBounds { index: KernelValueRef, extent: ExtentExpr },
    /// `0 <= start <= end <= extent`.
    RangeInBounds { start: KernelValueRef, end: KernelValueRef, extent: ExtentExpr },
    /// `value != 0` at `dtype`.
    DivisorNonZero { value: KernelValueRef, dtype: DType },
    /// `!(lhs == i32::MIN && rhs == -1)` (the divisor's zero test is a
    /// separate obligation).
    SignedDivisionNoOverflow { lhs: KernelValueRef, rhs: KernelValueRef },
    /// `0 <= value < 32`.
    ShiftInRange { value: KernelValueRef },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum BarrierScope {
    Subgroup,
    Workgroup,
}

/// How an `Atomic` update is realized, chosen from the block's participant
/// policy: one participant owns the domain (`Serialized`), or several
/// participants update concurrently (`Device`, 32-bit elements only).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum AtomicMode {
    /// One participant owns the domain: plain load/combine/round/store.
    Serialized,
    /// Device atomic: native fetch-add/max/min for 32-bit ints, f32 add
    /// native where the target has it, otherwise compare-exchange on the
    /// bits (float max/min ignore a NaN operand).
    Device,
}

/// One axis of an in-block slice, in kernel values.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum KernelSliceAxis {
    Full,
    Point(KernelValueRef),
    /// `start` absent means the axis start; the end never affects addressing.
    Range { start: Option<KernelValueRef> },
}

/// One in-block view step over a place, in the place's view coordinates.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum KernelViewStep {
    /// Row-major reinterpretation of `source_shape` as the step's result.
    Reshape { source_shape: Vec<ExtentExpr> },
    /// Result axis `i` is source axis `permutation[i]`.
    Transpose { permutation: Vec<u32> },
    Slice { axes: Vec<KernelSliceAxis> },
}

/// The composed in-block view chain of one tensor intrinsic operand: the
/// place's view coordinates transformed by `steps`, outermost source first.
/// Empty steps mean the place itself.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct KernelViewChain {
    pub steps: Vec<KernelViewStep>,
}

/// One typed operand of a backend intrinsic.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum IntrinsicOperand {
    /// A kernel scalar or capability value.
    Value(KernelValueRef),
    /// A whole tensor of type `ty`: `place` viewed through `view`.
    Tensor { place: KernelPlaceRef, ty: TensorType, view: KernelViewChain },
}

/// The typed result of a backend intrinsic.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum IntrinsicResult {
    Void,
    /// A kernel scalar or capability value the intrinsic defines.
    Ssa { id: KernelSsaId, ty: KernelValueType },
    /// A whole tensor the intrinsic writes into `place` (view coordinates).
    Tensor { place: KernelPlaceRef, ty: TensorType },
}

/// The closed core kernel operation algebra. Every backend encoder matches
/// it exhaustively. Control forms nest their bodies, which may contain
/// dialect intrinsics (`I`). See the module documentation for the exact
/// semantics of every variant.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CoreKernelOp<I> {
    Const { into: KernelSsaId, value: TypedConstant },
    /// The retained runtime value of one runtime extent, as `i32`.
    RuntimeExtent { into: KernelSsaId, extent: RuntimeExtentId },
    Unary { into: KernelSsaId, op: UnaryOp, operand: KernelValueRef, dtype: DType },
    Binary { into: KernelSsaId, op: BinaryOp, left: KernelValueRef, right: KernelValueRef, dtype: DType },
    /// Comparison at the operand `dtype`; the result is `bool`.
    Compare { into: KernelSsaId, op: RelOp, left: KernelValueRef, right: KernelValueRef, dtype: DType },
    Math { into: KernelSsaId, op: MathOp, operands: Vec<KernelValueRef>, dtype: DType },
    /// `into := a * b + c` rounded once at `dtype`.
    Fma { into: KernelSsaId, a: KernelValueRef, b: KernelValueRef, c: KernelValueRef, dtype: DType },
    Cast { into: KernelSsaId, operand: KernelValueRef, from: DType, to: DType },
    Select { into: KernelSsaId, condition: KernelValueRef, then_value: KernelValueRef, else_value: KernelValueRef, dtype: DType },
    /// `into: i32 := table[index]` for a `u32` code `index < table.len()`.
    TableLookup { into: KernelSsaId, index: KernelValueRef, table: Vec<i32> },
    /// Load one element of a dense plane at the given coordinates (in the
    /// place's view coordinates).
    Load { into: KernelSsaId, place: KernelPlaceRef, coords: Vec<KernelValueRef>, dtype: DType },
    /// Store one element.
    Store { place: KernelPlaceRef, coords: Vec<KernelValueRef>, value: KernelValueRef, dtype: DType },
    /// Read entry `entry` of the group of the logical element at `coords` in
    /// plane `plane` of representation `repr`; `dtype` is the plane's
    /// storage dtype (a packed plane yields the raw code zero-extended).
    PackedPlaneRead { into: KernelSsaId, place: KernelPlaceRef, coords: Vec<KernelValueRef>, repr: String, plane: PlaneField, entry: u32, dtype: DType },
    /// Load one storage element of plane `plane`: `coords` are the place's
    /// view coordinates with the packing-axis coordinate replaced by the
    /// storage-element ordinal within the plane row.
    PlaneLoad { into: KernelSsaId, place: KernelPlaceRef, coords: Vec<KernelValueRef>, repr: String, plane: PlaneField, dtype: DType },
    /// Store one storage element of plane `plane` (coordinates as `PlaneLoad`).
    PlaneStore { place: KernelPlaceRef, coords: Vec<KernelValueRef>, repr: String, plane: PlaneField, value: KernelValueRef, dtype: DType },
    /// Atomic read-modify-write of one element under `mode`.
    Atomic { place: KernelPlaceRef, coords: Vec<KernelValueRef>, op: AtomicOp, value: KernelValueRef, dtype: DType, mode: AtomicMode },
    /// Ascending serial loop over `[start, end)` with a block-local binder.
    Repeat { binder: KernelSsaId, start: KernelValueRef, end: KernelValueRef, body: Vec<KernelOp<I>> },
    /// Ordered scalar carry through the enclosing `Repeat` (listed at the
    /// head of its body): `current` is rebound each visit from `initial`
    /// then `update`; `result` is the final value after the loop.
    Carry { initial: KernelValueRef, current: KernelSsaId, update: KernelValueRef, result: KernelSsaId, ty: KernelValueType },
    Branch { condition: KernelValueRef, then_body: Vec<KernelOp<I>>, else_body: Vec<KernelOp<I>>, joins: Vec<KernelJoin> },
    /// Serial reduction fold over axis `axis` of `place` at the fixed
    /// `coords` of every other axis (in axis order), under `schema`.
    Fold { into: KernelSsaId, op: ReduceOp, place: KernelPlaceRef, axis: u32, coords: Vec<KernelValueRef>, schema: ReduceSchema, shape: TensorType },
    /// Guarded safety check writing the status field on failure; every op in
    /// `guarded` executes only when the predicate holds.
    Check { obligation: ObligationRef, predicate: CheckPredicate, status: StatusFieldTemplateId, guarded: Vec<KernelOp<I>> },
    /// Publish a kernel value to an external scalar destination.
    Publish { value: KernelValueRef, output: KernelOutputId },
    Barrier { scope: BarrierScope },
    /// Grid-wide synchronization of a `GridCooperative` block (a defect in
    /// any other block).
    GridBarrier,
}

/// One branch join inside a kernel block.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct KernelJoin {
    pub then_value: KernelValueRef,
    pub else_value: KernelValueRef,
    pub joined: KernelSsaId,
    pub ty: KernelValueType,
}

/// One kernel operation: core or backend intrinsic.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum KernelOp<I> {
    Core(CoreKernelOp<I>),
    Intrinsic(I),
}

/// One kernel SSA declaration.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct KernelSsaDecl {
    pub id: KernelSsaId,
    pub ty: KernelValueType,
}

impl seismic_lang::logical::IdIndex for KernelSsaId {
    fn from_index(index: usize) -> Self {
        KernelSsaId(index as u32)
    }
    fn index(self) -> usize {
        self.0 as usize
    }
}

impl<I> CoreKernelOp<I> {
    /// The SSA values this op itself defines (nested bodies excluded).
    pub fn defines(&self) -> Vec<KernelSsaId> {
        match self {
            CoreKernelOp::Const { into, .. }
            | CoreKernelOp::RuntimeExtent { into, .. }
            | CoreKernelOp::Unary { into, .. }
            | CoreKernelOp::Binary { into, .. }
            | CoreKernelOp::Compare { into, .. }
            | CoreKernelOp::Math { into, .. }
            | CoreKernelOp::Fma { into, .. }
            | CoreKernelOp::Cast { into, .. }
            | CoreKernelOp::Select { into, .. }
            | CoreKernelOp::TableLookup { into, .. }
            | CoreKernelOp::Load { into, .. }
            | CoreKernelOp::PackedPlaneRead { into, .. }
            | CoreKernelOp::PlaneLoad { into, .. }
            | CoreKernelOp::Fold { into, .. } => vec![*into],
            CoreKernelOp::Repeat { binder, .. } => vec![*binder],
            CoreKernelOp::Carry { current, result, .. } => vec![*current, *result],
            CoreKernelOp::Branch { joins, .. } => joins.iter().map(|join| join.joined).collect(),
            CoreKernelOp::Store { .. }
            | CoreKernelOp::PlaneStore { .. }
            | CoreKernelOp::Atomic { .. }
            | CoreKernelOp::Check { .. }
            | CoreKernelOp::Publish { .. }
            | CoreKernelOp::Barrier { .. }
            | CoreKernelOp::GridBarrier => Vec::new(),
        }
    }

    /// The values this op itself reads (nested bodies excluded).
    pub fn uses(&self) -> Vec<KernelValueRef> {
        match self {
            CoreKernelOp::Const { .. }
            | CoreKernelOp::RuntimeExtent { .. }
            | CoreKernelOp::Barrier { .. }
            | CoreKernelOp::GridBarrier => Vec::new(),
            CoreKernelOp::Unary { operand, .. } | CoreKernelOp::Cast { operand, .. } => vec![*operand],
            CoreKernelOp::Binary { left, right, .. } | CoreKernelOp::Compare { left, right, .. } => {
                vec![*left, *right]
            }
            CoreKernelOp::Math { operands, .. } => operands.clone(),
            CoreKernelOp::Fma { a, b, c, .. } => vec![*a, *b, *c],
            CoreKernelOp::Select {
                condition,
                then_value,
                else_value,
                ..
            } => vec![*condition, *then_value, *else_value],
            CoreKernelOp::TableLookup { index, .. } => vec![*index],
            CoreKernelOp::Load { coords, .. }
            | CoreKernelOp::PackedPlaneRead { coords, .. }
            | CoreKernelOp::PlaneLoad { coords, .. }
            | CoreKernelOp::Fold { coords, .. } => coords.clone(),
            CoreKernelOp::Store { coords, value, .. }
            | CoreKernelOp::PlaneStore { coords, value, .. }
            | CoreKernelOp::Atomic { coords, value, .. } => {
                let mut uses = coords.clone();
                uses.push(*value);
                uses
            }
            CoreKernelOp::Repeat { start, end, .. } => vec![*start, *end],
            CoreKernelOp::Carry { initial, update, .. } => vec![*initial, *update],
            CoreKernelOp::Branch { condition, joins, .. } => {
                let mut uses = vec![*condition];
                for join in joins {
                    uses.push(join.then_value);
                    uses.push(join.else_value);
                }
                uses
            }
            CoreKernelOp::Check { predicate, .. } => match predicate {
                CheckPredicate::IndexInBounds { index, .. } => vec![*index],
                CheckPredicate::RangeInBounds { start, end, .. } => vec![*start, *end],
                CheckPredicate::DivisorNonZero { value, .. } => vec![*value],
                CheckPredicate::SignedDivisionNoOverflow { lhs, rhs } => vec![*lhs, *rhs],
                CheckPredicate::ShiftInRange { value } => vec![*value],
            },
            CoreKernelOp::Publish { value, .. } => vec![*value],
        }
    }

    /// The places this op itself addresses (nested bodies excluded).
    pub fn places(&self) -> Vec<KernelPlaceRef> {
        match self {
            CoreKernelOp::Load { place, .. }
            | CoreKernelOp::Store { place, .. }
            | CoreKernelOp::PackedPlaneRead { place, .. }
            | CoreKernelOp::PlaneLoad { place, .. }
            | CoreKernelOp::PlaneStore { place, .. }
            | CoreKernelOp::Atomic { place, .. }
            | CoreKernelOp::Fold { place, .. } => vec![*place],
            CoreKernelOp::Const { .. }
            | CoreKernelOp::RuntimeExtent { .. }
            | CoreKernelOp::Unary { .. }
            | CoreKernelOp::Binary { .. }
            | CoreKernelOp::Compare { .. }
            | CoreKernelOp::Math { .. }
            | CoreKernelOp::Fma { .. }
            | CoreKernelOp::Cast { .. }
            | CoreKernelOp::Select { .. }
            | CoreKernelOp::TableLookup { .. }
            | CoreKernelOp::Repeat { .. }
            | CoreKernelOp::Carry { .. }
            | CoreKernelOp::Branch { .. }
            | CoreKernelOp::Check { .. }
            | CoreKernelOp::Publish { .. }
            | CoreKernelOp::Barrier { .. }
            | CoreKernelOp::GridBarrier => Vec::new(),
        }
    }

    /// The nested bodies of a control op, in execution order.
    pub fn bodies(&self) -> Vec<&[KernelOp<I>]> {
        match self {
            CoreKernelOp::Repeat { body, .. } => vec![body.as_slice()],
            CoreKernelOp::Branch {
                then_body, else_body, ..
            } => vec![then_body.as_slice(), else_body.as_slice()],
            CoreKernelOp::Check { guarded, .. } => vec![guarded.as_slice()],
            CoreKernelOp::Const { .. }
            | CoreKernelOp::RuntimeExtent { .. }
            | CoreKernelOp::Unary { .. }
            | CoreKernelOp::Binary { .. }
            | CoreKernelOp::Compare { .. }
            | CoreKernelOp::Math { .. }
            | CoreKernelOp::Fma { .. }
            | CoreKernelOp::Cast { .. }
            | CoreKernelOp::Select { .. }
            | CoreKernelOp::TableLookup { .. }
            | CoreKernelOp::Load { .. }
            | CoreKernelOp::Store { .. }
            | CoreKernelOp::PackedPlaneRead { .. }
            | CoreKernelOp::PlaneLoad { .. }
            | CoreKernelOp::PlaneStore { .. }
            | CoreKernelOp::Atomic { .. }
            | CoreKernelOp::Carry { .. }
            | CoreKernelOp::Fold { .. }
            | CoreKernelOp::Publish { .. }
            | CoreKernelOp::Barrier { .. }
            | CoreKernelOp::GridBarrier => Vec::new(),
        }
    }
}

impl<I> KernelOp<I> {
    /// Total definition/use listing of one op (nested bodies excluded):
    /// core ops answer structurally, intrinsics through the dialect.
    pub fn references(&self, intrinsic: &dyn Fn(&I) -> IntrinsicReferences) -> IntrinsicReferences {
        match self {
            KernelOp::Core(op) => IntrinsicReferences {
                uses: op.uses(),
                defines: op.defines(),
                places: op.places(),
            },
            KernelOp::Intrinsic(op) => intrinsic(op),
        }
    }

    /// The nested bodies of this op (empty for every non-control op and for
    /// every intrinsic).
    pub fn bodies(&self) -> Vec<&[KernelOp<I>]> {
        match self {
            KernelOp::Core(op) => op.bodies(),
            KernelOp::Intrinsic(_) => Vec::new(),
        }
    }
}

/// A closed, typed kernel block. No public constructor; sealed by
/// `KernelFormer::form` by exact definition/use and publication-set
/// equality.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ClosedKernelBlock<I> {
    interface: ClosedKernelInterface,
    ssa: IdVec<KernelSsaId, KernelSsaDecl>,
    ops: NonEmpty<KernelOp<I>>,
    status_fields: Vec<(StatusFieldTemplateId, ObligationRef)>,
}

impl<I> ClosedKernelBlock<I> {
    pub(crate) fn seal(
        interface: ClosedKernelInterface,
        ssa: IdVec<KernelSsaId, KernelSsaDecl>,
        ops: NonEmpty<KernelOp<I>>,
        status_fields: Vec<(StatusFieldTemplateId, ObligationRef)>,
    ) -> ClosedKernelBlock<I> {
        ClosedKernelBlock {
            interface,
            ssa,
            ops,
            status_fields,
        }
    }

    pub fn interface(&self) -> &ClosedKernelInterface {
        &self.interface
    }
    pub fn ssa(&self) -> &IdVec<KernelSsaId, KernelSsaDecl> {
        &self.ssa
    }
    pub fn ops(&self) -> &NonEmpty<KernelOp<I>> {
        &self.ops
    }
    pub fn status_fields(&self) -> &[(StatusFieldTemplateId, ObligationRef)] {
        &self.status_fields
    }

    /// Pre-order walk over every op, nested bodies included.
    pub fn walk(&self, f: &mut dyn FnMut(&KernelOp<I>)) {
        fn visit<I>(ops: &[KernelOp<I>], f: &mut dyn FnMut(&KernelOp<I>)) {
            for op in ops {
                f(op);
                for body in op.bodies() {
                    visit(body, f);
                }
            }
        }
        visit(self.ops.as_slice(), f);
    }

    /// One cost unit per static op site (nested sites included once; M1
    /// weights sites by the trip counts of their enclosing `Repeat`s and by
    /// the block's iteration map).
    pub fn cost_units<D: ExecutableDialect<Intrinsic = I>>(&self) -> Vec<CostUnit> {
        let mut units = Vec::new();
        self.walk(&mut |op| match op {
            KernelOp::Core(core) => match core {
                CoreKernelOp::Const { .. }
                | CoreKernelOp::RuntimeExtent { .. }
                | CoreKernelOp::Unary { .. }
                | CoreKernelOp::Binary { .. }
                | CoreKernelOp::Compare { .. }
                | CoreKernelOp::Cast { .. }
                | CoreKernelOp::Select { .. }
                | CoreKernelOp::TableLookup { .. }
                | CoreKernelOp::Repeat { .. }
                | CoreKernelOp::Carry { .. }
                | CoreKernelOp::Branch { .. }
                | CoreKernelOp::Check { .. }
                | CoreKernelOp::Publish { .. } => units.push(CostUnit::Scalar),
                CoreKernelOp::Math { op, dtype, .. } => units.push(CostUnit::Math {
                    op: *op,
                    dtype: *dtype,
                }),
                CoreKernelOp::Fma { dtype, .. } => units.push(CostUnit::Math {
                    op: MathOp::Fma,
                    dtype: *dtype,
                }),
                CoreKernelOp::Load { dtype, .. } => units.push(CostUnit::Load { dtype: *dtype }),
                CoreKernelOp::Store { dtype, .. } => units.push(CostUnit::Store { dtype: *dtype }),
                CoreKernelOp::PackedPlaneRead { repr, dtype, .. }
                | CoreKernelOp::PlaneLoad { repr, dtype, .. }
                | CoreKernelOp::PlaneStore { repr, dtype, .. } => units.push(CostUnit::PlaneAccess {
                    repr: repr.clone(),
                    dtype: *dtype,
                }),
                CoreKernelOp::Atomic { dtype, .. } => units.push(CostUnit::Atomic { dtype: *dtype }),
                CoreKernelOp::Fold { op, schema, .. } => units.push(CostUnit::Fold {
                    op: *op,
                    dtype: schema.accumulator,
                }),
                CoreKernelOp::Barrier { .. } | CoreKernelOp::GridBarrier => units.push(CostUnit::Barrier),
            },
            KernelOp::Intrinsic(intrinsic) => {
                units.push(CostUnit::Intrinsic(D::intrinsic_consequences(intrinsic).capability))
            }
        });
        units
    }
}

/// Which kernel values, places, and status fields one intrinsic references.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct IntrinsicReferences {
    pub uses: Vec<KernelValueRef>,
    pub defines: Vec<KernelSsaId>,
    pub places: Vec<KernelPlaceRef>,
}

/// The exact typed consequences of one intrinsic: hard resources, numerical
/// transfer, and the capability signature it requires. Cost is not here.
#[derive(Clone, Debug, PartialEq)]
pub struct IntrinsicConsequences {
    pub capability: IntrinsicId,
    pub numerical: crate::numerics::NumericalTransfer,
    pub private_bytes: u64,
    pub workgroup_bytes: u64,
    pub required_subgroup_width: Option<u32>,
}

/// A cost unit the backend cost model prices: one core op class or one
/// intrinsic family.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CostUnit {
    Scalar,
    Load { dtype: DType },
    Store { dtype: DType },
    /// One representation-plane entry or storage-element access.
    PlaneAccess { repr: String, dtype: DType },
    Math { op: MathOp, dtype: DType },
    Atomic { dtype: DType },
    Fold { op: ReduceOp, dtype: DType },
    Barrier,
    Intrinsic(IntrinsicId),
}

/// The typed intrinsic lowering families a backend authorizes.
///
/// Totality contract (B1): the catalog is constructed from the target's
/// `EffectiveTargetProfile::effective_signatures`, and every use K1 can
/// present is authorized: (1) a checked `PrimitiveOp::Capability(id)`
/// application whose `id` is an effective signature (S1 declines every
/// proposal otherwise), with operands typed exactly as the use's
/// `IntrinsicUse::arguments` (scalars and capability values as `Value`,
/// tensors as `Tensor` places with their in-block view chain) and the result
/// typed as `IntrinsicUse::result`; and (2) a reduction or capability node of
/// a block whose `AlgorithmChoice::Intrinsic(id)` or cooperative
/// `ParticipantPolicy::Cooperative { family: id, .. }` names an effective
/// signature the backend's own mapping rule proposed, with the node's
/// operand and result types. `lower` is total over authorized uses; a use it
/// cannot map is a B1 bug, and the implementation may panic with owner
/// attribution for genuinely impossible arms. It never returns a placeholder.
pub trait IntrinsicCatalog<D: ExecutableDialect> {
    /// Lower one authorized capability use with typed operands and result.
    fn lower(
        &self,
        intrinsic: &IntrinsicUse,
        operands: &[IntrinsicOperand],
        result: IntrinsicResult,
    ) -> D::Intrinsic;
}

/// The typed descriptor of one residence plane: every fact a layout
/// function needs, with no optional member. A `Dense` plane carries its
/// element dtype directly, so a dense plane over a `Param`/`Repr` element
/// (a schema-less dense plane) is unrepresentable — `PlaneRef::of` rejects
/// the pair as a defect instead of letting a backend substitute a dtype.
/// A representation plane carries the full registered plane schema.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PlaneRef {
    /// One dense plane of element dtype `elem`.
    Dense { elem: DType },
    /// One plane of a registered representation.
    Repr { plane: repr::Plane },
}

impl PlaneRef {
    /// The dtype of one storage element of the plane.
    pub fn storage_dtype(&self) -> DType {
        match self {
            PlaneRef::Dense { elem } => *elem,
            PlaneRef::Repr { plane } => plane.dtype(),
        }
    }

    /// The descriptor of one declared residence plane over a tensor type.
    ///
    /// Total over the admissible pairs; the inadmissible pairs are named
    /// defects: a dense plane over a packed element (the residence must
    /// declare the representation's planes), a dense plane over an element
    /// parameter that survived specialization, an unregistered
    /// representation, and a plane ordinal the representation does not
    /// declare.
    pub fn of(tensor: &TensorType, plane: &StoragePlane) -> Result<PlaneRef, CompilerDefect> {
        match plane {
            StoragePlane::Dense => match &tensor.elem {
                Elem::Dtype(elem) => Ok(PlaneRef::Dense { elem: *elem }),
                Elem::Repr(name) => Err(CompilerDefect::new(
                    Package::D1,
                    format!(
                        "a dense plane is declared over the packed element `{name}`; \
                         a packed element requires its representation's planes"
                    ),
                )),
                Elem::Param(name) => Err(CompilerDefect::new(
                    Package::W1,
                    format!("element parameter `{name}` survived specialization"),
                )),
            },
            StoragePlane::Representation { name, ordinal } => {
                let Some(representation) = repr::lookup(name) else {
                    return Err(CompilerDefect::new(
                        Package::L1,
                        format!("unknown representation `{name}`"),
                    ));
                };
                match representation.planes().get(*ordinal as usize) {
                    Some(plane) => Ok(PlaneRef::Repr { plane: plane.clone() }),
                    None => Err(CompilerDefect::new(
                        Package::D1,
                        format!("representation `{name}` declares no plane {ordinal}"),
                    )),
                }
            }
        }
    }
}

/// The sealed backend dialect contract.
pub trait ExecutableDialect: sealed::Sealed + 'static {
    /// Closed intrinsic enum; encoded exhaustively, never with a wildcard.
    type Intrinsic: Clone + std::fmt::Debug + PartialEq + Eq;
    type LayoutTemplate: Clone + std::fmt::Debug + PartialEq + Eq;
    type ResolvedLayout: Clone + std::fmt::Debug + PartialEq + Eq;

    fn intrinsic_references(op: &Self::Intrinsic) -> IntrinsicReferences;
    fn intrinsic_consequences(op: &Self::Intrinsic) -> IntrinsicConsequences;
    /// Layout of a public (root ABI) residence plane.
    ///
    /// Totality contract: `plane` is a required typed descriptor that
    /// supplies every fact the layout needs — the element dtype of a dense
    /// plane, the full schema of a representation plane. Data-reachable
    /// domain conditions (an extent outside the i64 size domain, an
    /// unresolved symbol, an inadmissible plane/tensor pair) are rejected
    /// upstream at the Result-returning front doors of D1, W1, and P1; a
    /// condition of that kind reaching the layout function is an invariant
    /// violation and may panic naming the owning package (D1, W1, P1, or B1
    /// as appropriate). A backend that cannot map an admissible descriptor
    /// onto its target's layout vocabulary is a B1 bug. It never substitutes
    /// a default dtype, extent, or size.
    fn public_layout(tensor: &TensorType, plane: PlaneRef) -> Self::LayoutTemplate;
    /// Layout of an internal residence plane, under the same totality
    /// contract as `public_layout`.
    fn internal_layout(tensor: &TensorType, plane: PlaneRef) -> Self::LayoutTemplate;
    /// Mechanical substitution under validated solved values. Infallible:
    /// the seal validated every value before substitution.
    fn resolve_layout(layout: &Self::LayoutTemplate, values: &crate::plan_space::SolvedValues) -> Self::ResolvedLayout;
}

pub mod sealed {
    pub trait Sealed {}
}

/// The sole constructor of `ClosedKernelBlock` (package K1).
pub struct KernelFormer;

impl KernelFormer {
    pub fn form<D: ExecutableDialect>(
        facts: &crate::occurrence::OccurrenceFacts<'_>,
        strategy: &crate::residence::RoutedStrategy,
        block: crate::ids::BlockId,
        catalog: &dyn IntrinsicCatalog<D>,
    ) -> Result<ClosedKernelBlock<D::Intrinsic>, crate::failure::CompilerDefect> {
        crate::formation::kernel::form::<D>(facts, strategy, block, catalog)
    }
}
