//! Typed kernel IR and `KernelBuilder` (spec §7).
//!
//! Handles are category- and representation-typed. Lexical construction:
//! a value created in a block is usable only in that block or a dominated
//! child; branches own their join and repeats own their carries, and both
//! close only with identical typed result schemas. Intrinsics are built from
//! registered typed signatures via [`TypedIntrinsic`]. Materialization is a
//! compiler operation performed by the implementation builder, never by
//! source.
//!
//! The IR is consumed by native compilers through the read surface of
//! [`Kernel`]. Public builder signatures preserve category; erased storage
//! lives behind `internals` (W4-owned).

use crate::identity::OwnerToken;
use crate::repr::{
    AtomicType, Bool, FloatType, Idx, IntegerType, NumericType, Representation, ScalarType,
    SignedType, VectorElement, WritableRepresentation, U32,
};
use crate::storage::{BufferViewId, LaunchLocalId, LaunchLocalKind};
use crate::target::Backend;
use seismic_lang::expr::NatExpr;
use seismic_lang::ids::IntrinsicId;
use seismic_lang::intrinsics::AtomicOp;
use std::fmt;
use std::marker::PhantomData;

pub mod ops;
mod vector;
pub use vector::VectorId;

// ---------------------------------------------------------------------------
// Handles
// ---------------------------------------------------------------------------

macro_rules! typed_handle {
    ($(#[$doc:meta])* $name:ident < $param:ident : $bound:path > , $prefix:literal) => {
        $(#[$doc])*
        #[derive(PartialEq, Eq, PartialOrd, Ord, Hash)]
        pub struct $name<$param: $bound> {
            owner: OwnerToken,
            kernel: u32,
            block: BlockId,
            index: u32,
            marker: PhantomData<$param>,
        }
        impl<$param: $bound> Clone for $name<$param> {
            fn clone(&self) -> Self {
                *self
            }
        }
        impl<$param: $bound> Copy for $name<$param> {}
        impl<$param: $bound> fmt::Debug for $name<$param> {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                write!(f, "{}#{}", $prefix, self.index)
            }
        }
        impl<$param: $bound> $name<$param> {
            pub(crate) fn new(owner: OwnerToken, kernel: u32, block: BlockId, index: u32) -> Self {
                Self { owner, kernel, block, index, marker: PhantomData }
            }
            pub(crate) fn index(self) -> u32 {
                self.index
            }
            pub(crate) fn owner(self) -> OwnerToken { self.owner }
            pub(crate) fn kernel(self) -> u32 { self.kernel }
            pub(crate) fn block(self) -> BlockId { self.block }
        }
    };
}

typed_handle!(/// One SSA scalar.
    ScalarId<T: ScalarType>, "scalar");
typed_handle!(/// A readable tensor place (global view or local) inside a kernel.
    ReadablePlaceId<R: Representation>, "readable");
typed_handle!(/// A writable tensor place inside a kernel.
    WritablePlaceId<R: Representation>, "writable");
typed_handle!(/// One physical plane of a packed place.
    PlaneId<R: Representation>, "plane");
typed_handle!(/// A computed (not yet materialized) tensor value inside a kernel.
    TensorId<R: Representation>, "tensor");

/// One kernel within an implementation's arena.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct KernelId {
    owner: OwnerToken,
    index: u32,
}

impl KernelId {
    pub(crate) fn new(owner: OwnerToken, index: u32) -> Self {
        Self { owner, index }
    }
    pub(crate) fn owner(self) -> OwnerToken {
        self.owner
    }
    pub fn ordinal(self) -> u32 {
        self.index
    }
    pub(crate) fn index(self) -> u32 {
        self.ordinal()
    }
}

/// A lexical block within a kernel. Blocks are created and closed only by
/// the builder.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct BlockId {
    owner: OwnerToken,
    kernel: u32,
    index: u32,
}

impl BlockId {
    pub(crate) fn new(owner: OwnerToken, kernel: u32, index: u32) -> Self {
        Self {
            owner,
            kernel,
            index,
        }
    }
    pub(crate) fn owner(self) -> OwnerToken {
        self.owner
    }
    pub(crate) fn kernel(self) -> u32 {
        self.kernel
    }
    pub fn ordinal(self) -> u32 {
        self.index
    }
    pub(crate) fn index(self) -> u32 {
        self.ordinal()
    }
}

/// A binding slot of a kernel: the position a global buffer view or scalar
/// argument occupies in the launch's argument table.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct BindingSlot {
    owner: OwnerToken,
    kernel: u32,
    index: u32,
}

impl BindingSlot {
    pub(crate) fn new(owner: OwnerToken, kernel: u32, index: u32) -> Self {
        Self {
            owner,
            kernel,
            index,
        }
    }
    pub(crate) fn owner(self) -> OwnerToken {
        self.owner
    }
    pub(crate) fn kernel(self) -> u32 {
        self.kernel
    }
    pub fn ordinal(self) -> u32 {
        self.index
    }
    pub(crate) fn index(self) -> u32 {
        self.ordinal()
    }
}

// ---------------------------------------------------------------------------
// Typed intrinsics
// ---------------------------------------------------------------------------

/// A backend intrinsic with typed arguments and result, defined by the
/// backend crate (W5) against a registered signature. The builder accepts
/// only `Args` and returns only `Result`, so an ill-typed intrinsic call does
/// not compile.
pub trait TypedIntrinsic<B: Backend>: 'static {
    /// The registered signature this intrinsic realizes.
    fn id() -> IntrinsicId;
    type Args;
    type Result: KernelValues;
    /// Lowers to the backend's intrinsic op given the erased handles. Called
    /// by the builder; not by factories.
    fn lower(args: &Self::Args, sink: &mut ops::IntrinsicSink<'_, B>) -> Self::Result;
    /// Launch-local resources the lowering requires (§8.3), as arena
    /// expressions so they enter the solver as constraints.
    fn resources(
        args: &Self::Args,
        arena: &mut seismic_lang::expr::ExprArena,
    ) -> ops::IntrinsicResources;
}

// ---------------------------------------------------------------------------
// Typed value tuples for joins and carries
// ---------------------------------------------------------------------------

/// A tuple of typed kernel values with one schema, for branch results and
/// loop carries. Implemented for `()`, `ScalarId<T>`, and tuples up to
/// eight.
pub trait KernelValues: values_sealed::Sealed + Copy + fmt::Debug {
    #[doc(hidden)]
    fn schema() -> ops::ValueSchema;
    #[doc(hidden)]
    fn erase(&self) -> Vec<ops::ErasedValue>;
    #[doc(hidden)]
    fn restore(values: &[ops::ErasedValue]) -> Self;
}

mod values_sealed {
    pub trait Sealed {}
}

// ---------------------------------------------------------------------------
// Builder
// ---------------------------------------------------------------------------

/// Builds one kernel. Obtained from `ImplementationBuilder::kernel`.
pub struct KernelBuilder<'a, B: Backend> {
    inner: internals::Builder<'a, B>,
}

/// Failure to close a kernel: the only construction-time failures are
/// lexical (a value used outside its dominating block) and schema mismatch
/// of a join/carry. Both are factory authoring bugs and are reported once at
/// close as a panic with the offending handle (§13.3, local invariant of the
/// private builder).
impl<'a, B: Backend> KernelBuilder<'a, B> {
    // ----- interface ---------------------------------------------------------

    /// Declares a readable global view argument.
    pub fn arg_readable<R: Representation>(&mut self, view: BufferViewId<R>) -> ReadablePlaceId<R> {
        self.inner.arg_readable(view)
    }
    /// Declares a writable global view argument.
    pub fn arg_writable<R: WritableRepresentation>(
        &mut self,
        view: BufferViewId<R>,
    ) -> WritablePlaceId<R> {
        self.inner.arg_writable(view)
    }
    /// Declares an immutable scalar argument bound from an arena `Nat`
    /// expression (invocation symbols, decisions, schedule slots).
    pub fn arg_nat(&mut self, expr: NatExpr) -> ScalarId<Idx> {
        self.inner.arg_nat(expr)
    }
    /// Declares a scalar argument bound from a call scalar parameter.
    pub fn arg_scalar<T: ScalarType>(
        &mut self,
        symbol: seismic_lang::expr::SymbolId,
    ) -> ScalarId<T> {
        self.inner.arg_scalar(symbol)
    }
    /// A scalar the kernel writes back to a schedule slot at completion
    /// (one participant elected by the builder rule).
    pub fn result_slot<T: ScalarType>(
        &mut self,
        slot: crate::schedule::ScalarSlotId<T>,
    ) -> WritableScalar<T> {
        self.inner.result_slot(slot)
    }

    // ----- launch-local storage ----------------------------------------------

    pub fn local<R: Representation>(
        &mut self,
        kind: LaunchLocalKind,
        extents: Vec<NatExpr>,
    ) -> LaunchLocalId<R> {
        self.inner.local(kind, extents)
    }
    pub fn local_readable<R: Representation>(
        &mut self,
        local: LaunchLocalId<R>,
    ) -> ReadablePlaceId<R> {
        self.inner.local_readable(local)
    }
    pub fn local_writable<R: WritableRepresentation>(
        &mut self,
        local: LaunchLocalId<R>,
    ) -> WritablePlaceId<R> {
        self.inner.local_writable(local)
    }
    /// A writable place is also readable, in program order.
    pub fn as_readable<R: Representation>(
        &mut self,
        place: WritablePlaceId<R>,
    ) -> ReadablePlaceId<R> {
        self.inner.as_readable(place)
    }

    // ----- geometry ------------------------------------------------------------

    pub fn workgroup_id(&mut self, axis: u8) -> ScalarId<Idx> {
        self.inner.workgroup_id(axis)
    }
    pub fn local_id(&mut self, axis: u8) -> ScalarId<Idx> {
        self.inner.local_id(axis)
    }
    pub fn global_id(&mut self, axis: u8) -> ScalarId<Idx> {
        self.inner.global_id(axis)
    }
    pub fn workgroup_size(&mut self, axis: u8) -> ScalarId<Idx> {
        self.inner.workgroup_size(axis)
    }
    pub fn grid_size(&mut self, axis: u8) -> ScalarId<Idx> {
        self.inner.grid_size(axis)
    }
    /// Subgroup lane index; requires the target's subgroup width to be
    /// present in the profile (the builder records the constraint).
    pub fn subgroup_lane(&mut self) -> ScalarId<Idx> {
        self.inner.subgroup_lane()
    }

    // ----- scalars -------------------------------------------------------------

    pub fn constant<T: ScalarType>(&mut self, value: T::Value) -> ScalarId<T> {
        self.inner.constant::<T>(value)
    }
    pub fn add<T: NumericType>(&mut self, a: ScalarId<T>, b: ScalarId<T>) -> ScalarId<T> {
        self.inner.binary(ops::BinaryOp::Add, a, b)
    }
    pub fn sub<T: NumericType>(&mut self, a: ScalarId<T>, b: ScalarId<T>) -> ScalarId<T> {
        self.inner.binary(ops::BinaryOp::Sub, a, b)
    }
    pub fn mul<T: NumericType>(&mut self, a: ScalarId<T>, b: ScalarId<T>) -> ScalarId<T> {
        self.inner.binary(ops::BinaryOp::Mul, a, b)
    }
    pub fn div<T: NumericType>(&mut self, a: ScalarId<T>, b: ScalarId<T>) -> ScalarId<T> {
        self.inner.binary(ops::BinaryOp::Div, a, b)
    }
    pub fn rem<T: IntegerType>(&mut self, a: ScalarId<T>, b: ScalarId<T>) -> ScalarId<T> {
        self.inner.binary(ops::BinaryOp::Rem, a, b)
    }
    pub fn min<T: NumericType>(&mut self, a: ScalarId<T>, b: ScalarId<T>) -> ScalarId<T> {
        self.inner.binary(ops::BinaryOp::Min, a, b)
    }
    pub fn max<T: NumericType>(&mut self, a: ScalarId<T>, b: ScalarId<T>) -> ScalarId<T> {
        self.inner.binary(ops::BinaryOp::Max, a, b)
    }
    pub fn bit<T: IntegerType>(
        &mut self,
        op: ops::BitOp,
        a: ScalarId<T>,
        b: ScalarId<T>,
    ) -> ScalarId<T> {
        self.inner.bit(op, a, b)
    }
    pub fn neg<T: SignedType>(&mut self, a: ScalarId<T>) -> ScalarId<T> {
        self.inner.unary(ops::UnaryOp::Neg, a)
    }
    /// Source-authored single-rounded fused multiply-add. Because this is the
    /// authored operation, it introduces no implementation deviation.
    pub fn fma<T: FloatType>(
        &mut self,
        a: ScalarId<T>,
        b: ScalarId<T>,
        c: ScalarId<T>,
    ) -> ScalarId<T> {
        self.inner.fma(a, b, c, false)
    }
    /// An implementation-selected contraction of separately authored
    /// multiply and add operations. This explicitly records the numerical
    /// deviation; factories must not use it for an authored FMA.
    pub fn contracted_fma<T: FloatType>(
        &mut self,
        a: ScalarId<T>,
        b: ScalarId<T>,
        c: ScalarId<T>,
    ) -> ScalarId<T> {
        self.inner.fma(a, b, c, true)
    }
    /// Registry math op with exact (reference) semantics.
    pub fn math<T: FloatType>(&mut self, op: ops::UnaryMathOp, a: ScalarId<T>) -> ScalarId<T> {
        let primitive = op.primitive();
        let precision = if primitive == seismic_lang::intrinsics::MathOp::ExpFast {
            ops::MathPrecision::Approximate
        } else {
            ops::MathPrecision::Exact
        };
        self.inner.math(primitive, a, precision)
    }
    /// Registry math op with an approximate backend sequence; admissible
    /// only when the implementation's numerical transfer records it.
    pub fn math_approximate<T: FloatType>(
        &mut self,
        op: ops::UnaryMathOp,
        a: ScalarId<T>,
    ) -> ScalarId<T> {
        self.inner
            .math(op.primitive(), a, ops::MathPrecision::Approximate)
    }
    /// Cast with registry rounding rules.
    pub fn cast<From: ScalarType, To: ScalarType>(&mut self, a: ScalarId<From>) -> ScalarId<To> {
        self.inner.cast(a)
    }
    pub fn cmp<T: ScalarType>(
        &mut self,
        op: ops::CmpOp,
        a: ScalarId<T>,
        b: ScalarId<T>,
    ) -> ScalarId<Bool> {
        self.inner.cmp(op, a, b)
    }
    pub fn select<T: ScalarType>(
        &mut self,
        cond: ScalarId<Bool>,
        a: ScalarId<T>,
        b: ScalarId<T>,
    ) -> ScalarId<T> {
        self.inner.select(cond, a, b)
    }
    pub fn logic(
        &mut self,
        op: ops::LogicOp,
        a: ScalarId<Bool>,
        b: ScalarId<Bool>,
    ) -> ScalarId<Bool> {
        self.inner.logic(op, a, b)
    }
    pub fn not(&mut self, a: ScalarId<Bool>) -> ScalarId<Bool> {
        self.inner.not(a)
    }
    pub fn index_from<T: IntegerType>(&mut self, a: ScalarId<T>) -> ScalarId<Idx> {
        self.inner.cast(a)
    }

    // ----- fixed-width vectors ------------------------------------------------

    pub fn vector_splat<T: VectorElement, const LANES: u16>(
        &mut self,
        value: ScalarId<T>,
    ) -> VectorId<T, LANES> {
        self.inner.vector_splat(value)
    }
    pub fn vector_add<T: NumericType + VectorElement, const LANES: u16>(
        &mut self,
        a: VectorId<T, LANES>,
        b: VectorId<T, LANES>,
    ) -> VectorId<T, LANES> {
        self.inner.vector_binary(ops::BinaryOp::Add, a, b)
    }
    pub fn vector_sub<T: NumericType + VectorElement, const LANES: u16>(
        &mut self,
        a: VectorId<T, LANES>,
        b: VectorId<T, LANES>,
    ) -> VectorId<T, LANES> {
        self.inner.vector_binary(ops::BinaryOp::Sub, a, b)
    }
    pub fn vector_mul<T: NumericType + VectorElement, const LANES: u16>(
        &mut self,
        a: VectorId<T, LANES>,
        b: VectorId<T, LANES>,
    ) -> VectorId<T, LANES> {
        self.inner.vector_binary(ops::BinaryOp::Mul, a, b)
    }
    pub fn vector_div<T: NumericType + VectorElement, const LANES: u16>(
        &mut self,
        a: VectorId<T, LANES>,
        b: VectorId<T, LANES>,
    ) -> VectorId<T, LANES> {
        self.inner.vector_binary(ops::BinaryOp::Div, a, b)
    }
    pub fn vector_rem<T: IntegerType + VectorElement, const LANES: u16>(
        &mut self,
        a: VectorId<T, LANES>,
        b: VectorId<T, LANES>,
    ) -> VectorId<T, LANES> {
        self.inner.vector_binary(ops::BinaryOp::Rem, a, b)
    }
    pub fn vector_min<T: NumericType + VectorElement, const LANES: u16>(
        &mut self,
        a: VectorId<T, LANES>,
        b: VectorId<T, LANES>,
    ) -> VectorId<T, LANES> {
        self.inner.vector_binary(ops::BinaryOp::Min, a, b)
    }
    pub fn vector_max<T: NumericType + VectorElement, const LANES: u16>(
        &mut self,
        a: VectorId<T, LANES>,
        b: VectorId<T, LANES>,
    ) -> VectorId<T, LANES> {
        self.inner.vector_binary(ops::BinaryOp::Max, a, b)
    }
    pub fn vector_neg<T: SignedType + VectorElement, const LANES: u16>(
        &mut self,
        value: VectorId<T, LANES>,
    ) -> VectorId<T, LANES> {
        self.inner.vector_unary(ops::UnaryOp::Neg, value)
    }
    pub fn vector_abs<T: SignedType + VectorElement, const LANES: u16>(
        &mut self,
        value: VectorId<T, LANES>,
    ) -> VectorId<T, LANES> {
        self.inner.vector_unary(ops::UnaryOp::Abs, value)
    }
    pub fn vector_bit<T: IntegerType + VectorElement, const LANES: u16>(
        &mut self,
        op: ops::BitOp,
        a: VectorId<T, LANES>,
        b: VectorId<T, LANES>,
    ) -> VectorId<T, LANES> {
        self.inner.vector_bit(op, a, b)
    }
    pub fn vector_fma<T: FloatType + VectorElement, const LANES: u16>(
        &mut self,
        a: VectorId<T, LANES>,
        b: VectorId<T, LANES>,
        c: VectorId<T, LANES>,
    ) -> VectorId<T, LANES> {
        self.inner.vector_fma(a, b, c)
    }
    pub fn vector_cast<From: VectorElement, To: VectorElement, const LANES: u16>(
        &mut self,
        value: VectorId<From, LANES>,
    ) -> VectorId<To, LANES> {
        self.inner.vector_cast(value)
    }
    pub fn vector_lane<T: VectorElement, const LANES: u16>(
        &mut self,
        vector: VectorId<T, LANES>,
        lane: u16,
    ) -> ScalarId<T> {
        self.inner.vector_lane(vector, lane)
    }
    /// Fixed ascending lane-order sum. This is an explicit reassociation
    /// relative to a scalar source chain and is recorded in numerical facts.
    pub fn vector_reduce_add<T: NumericType + VectorElement, const LANES: u16>(
        &mut self,
        vector: VectorId<T, LANES>,
    ) -> ScalarId<T> {
        self.inner.vector_reduce_add(vector)
    }

    // ----- memory --------------------------------------------------------------

    /// Reads one element, decoded to the representation's element type.
    pub fn read<R: Representation>(
        &mut self,
        place: ReadablePlaceId<R>,
        index: &[ScalarId<Idx>],
    ) -> ScalarId<R::Element> {
        self.inner.read(place, index)
    }
    /// Reads consecutive logical elements along `axis`, zeroing every lane
    /// whose ordinal is not less than `active`.
    pub fn vector_read<R: Representation, const LANES: u16>(
        &mut self,
        place: ReadablePlaceId<R>,
        index: &[ScalarId<Idx>],
        axis: u32,
        active: ScalarId<Idx>,
    ) -> VectorId<R::Element, LANES> {
        self.inner.vector_read(place, index, axis, active)
    }
    /// Writes consecutive logical elements along `axis`; lanes greater than
    /// or equal to `active` have no effect.
    pub fn vector_write<R: WritableRepresentation, const LANES: u16>(
        &mut self,
        place: WritablePlaceId<R>,
        index: &[ScalarId<Idx>],
        axis: u32,
        active: ScalarId<Idx>,
        value: VectorId<R::Element, LANES>,
    ) {
        self.inner.vector_write(place, index, axis, active, value)
    }
    pub fn write<R: WritableRepresentation>(
        &mut self,
        place: WritablePlaceId<R>,
        index: &[ScalarId<Idx>],
        value: ScalarId<R::Element>,
    ) {
        self.inner.write(place, index, value)
    }
    /// Reads a raw word of one plane of a packed place.
    pub fn read_plane<R: Representation>(
        &mut self,
        plane: PlaneId<R>,
        index: &[ScalarId<Idx>],
    ) -> ScalarId<U32> {
        self.inner.read_plane(plane, index)
    }
    pub fn plane<R: Representation>(
        &mut self,
        place: ReadablePlaceId<R>,
        plane: u32,
    ) -> PlaneId<R> {
        self.inner.plane(place, plane)
    }
    /// Extent of one axis of a place, as bound at launch.
    pub fn extent<R: Representation>(
        &mut self,
        place: ReadablePlaceId<R>,
        axis: u32,
    ) -> ScalarId<Idx> {
        self.inner.extent(place, axis)
    }
    pub fn atomic<R: WritableRepresentation>(
        &mut self,
        op: AtomicOp,
        place: WritablePlaceId<R>,
        index: &[ScalarId<Idx>],
        value: ScalarId<R::Element>,
    ) where
        R::Element: AtomicType,
    {
        self.inner.atomic(op, place, index, value)
    }
    pub fn store_slot<T: ScalarType>(&mut self, slot: WritableScalar<T>, value: ScalarId<T>) {
        self.inner.store_slot(slot, value)
    }

    // ----- synchronization -----------------------------------------------------

    pub fn workgroup_barrier(&mut self) {
        self.inner.barrier(ops::BarrierScope::Workgroup)
    }
    pub fn subgroup_barrier(&mut self) {
        self.inner.barrier(ops::BarrierScope::Subgroup)
    }

    // ----- intrinsics ----------------------------------------------------------

    pub fn intrinsic<I: TypedIntrinsic<B>>(&mut self, args: I::Args) -> I::Result {
        self.inner.intrinsic::<I>(args)
    }

    // ----- control -------------------------------------------------------------

    /// Structured branch. Both arms return the same typed schema; the join is
    /// owned by the branch.
    pub fn branch<V: KernelValues>(
        &mut self,
        cond: ScalarId<Bool>,
        then: impl FnOnce(&mut KernelBuilder<'_, B>) -> V,
        otherwise: impl FnOnce(&mut KernelBuilder<'_, B>) -> V,
    ) -> V {
        self.inner.branch(cond, then, otherwise)
    }

    /// Structured counted loop with typed carries. `body` receives the
    /// binder and the carried values and returns the next carried values.
    pub fn repeat<V: KernelValues>(
        &mut self,
        start: ScalarId<Idx>,
        end: ScalarId<Idx>,
        initial: V,
        body: impl FnOnce(&mut KernelBuilder<'_, B>, ScalarId<Idx>, V) -> V,
    ) -> V {
        self.inner.repeat(start, end, initial, body)
    }

    /// Closes the kernel. Requires every declared result slot to be
    /// written on every path (checked structurally).
    pub fn close(self) -> KernelId {
        self.inner.close()
    }
}

/// A writable scalar result slot handle inside a kernel.
#[derive(PartialEq, Eq, Hash)]
pub struct WritableScalar<T: ScalarType> {
    owner: OwnerToken,
    kernel: u32,
    index: u32,
    marker: PhantomData<T>,
}
impl<T: ScalarType> Clone for WritableScalar<T> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<T: ScalarType> Copy for WritableScalar<T> {}
impl<T: ScalarType> fmt::Debug for WritableScalar<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "slot#{}", self.index)
    }
}

// ---------------------------------------------------------------------------
// Read surface for native compilers
// ---------------------------------------------------------------------------

/// The arena of every kernel of one implementation (or one frozen plan).
#[derive(Debug)]
pub struct KernelArena<B: Backend> {
    inner: internals::Arena<B>,
}

impl<B: Backend> KernelArena<B> {
    pub(crate) fn retained_bytes(&self) -> usize {
        self.inner.retained_bytes()
    }
    pub fn kernels(&self) -> impl Iterator<Item = (KernelId, &Kernel<B>)> + '_ {
        self.inner.kernels()
    }
    pub fn kernel(&self, id: KernelId) -> &Kernel<B> {
        self.inner.kernel(id)
    }
}

/// One closed kernel, read by native compilers. Every reference inside is
/// dense and in-bounds by construction.
#[derive(Debug)]
pub struct Kernel<B: Backend> {
    inner: internals::KernelData<B>,
}

impl<B: Backend> Kernel<B> {
    pub fn interface(&self) -> &ops::KernelInterface {
        self.inner.interface()
    }
    pub fn locals(&self) -> &[crate::storage::LocalAllocation] {
        self.inner.locals()
    }
    pub fn root(&self) -> BlockId {
        self.inner.root()
    }
    pub fn block(&self, id: BlockId) -> &ops::Block<B> {
        self.inner.block(id)
    }
    pub fn value_type(&self, value: ops::ErasedValue) -> ops::ValueType {
        self.inner.value_type(value)
    }
    pub fn addressable_resource(
        &self,
        handle: ops::AddressableResourceHandle,
    ) -> &ops::AddressableResourceLease {
        assert_eq!(
            handle.owner, self.inner.owner,
            "resource handle belongs to another implementation"
        );
        assert_eq!(
            handle.kernel, self.inner.kernel,
            "resource handle belongs to another kernel"
        );
        self.inner
            .addressable_resources()
            .get(handle.lease as usize)
            .expect("resource handle is outside its closed kernel")
    }
    pub fn addressable_resources(&self) -> &[ops::AddressableResourceLease] {
        self.inner.addressable_resources()
    }
    pub fn closed_addressable_resource(
        &self,
        handle: ops::AddressableResourceHandle,
        emission: &crate::target::KernelEmissionLayout,
    ) -> ops::ClosedAddressableResource {
        let lease = self.addressable_resource(handle);
        let layout = emission
            .addressable_resources
            .get(handle.ordinal() as usize)
            .expect("resource handle has no emission layout");
        assert_eq!(
            layout.handle, handle,
            "resource emission ordinal differs from its handle"
        );
        ops::ClosedAddressableResource {
            handle,
            class_id: lease.class_id,
            class: lease.class.clone(),
            offset_word: layout.words.offset_units,
            units_word: layout.words.units,
            alignment_units: lease.alignment_units,
            lifetime: lease.lifetime,
        }
    }
    pub fn closed_value(&self, value: ops::ErasedValue) -> ops::ClosedValue {
        ops::ClosedValue {
            value,
            ty: self.inner.value_type(value),
        }
    }
    pub fn closed_place(
        &self,
        place: ops::PlaceRef,
        emission: &crate::target::KernelEmissionLayout,
    ) -> ops::ClosedPlace {
        match place {
            ops::PlaceRef::Global { slot } => {
                let binding = self
                    .interface()
                    .bindings
                    .get(slot.ordinal() as usize)
                    .expect("closed global place has no interface binding");
                assert_eq!(
                    binding.slot, slot,
                    "closed global place ordinal differs from its binding"
                );
                ops::ClosedPlace {
                    raw: place,
                    kind: ops::ClosedPlaceKind::Global {
                        slot,
                        access: binding.access,
                        buffer_ordinal: slot.ordinal(),
                    },
                    representation: binding.view.representation,
                    rank: binding.rank,
                    extents: binding.extents.clone(),
                    geometry: emission.bindings[slot.ordinal() as usize].geometry.clone(),
                    words: ops::ClosedPlaceWords::Binding(
                        emission.bindings[slot.ordinal() as usize].words,
                    ),
                    realization: None,
                }
            }
            ops::PlaceRef::Local { index } => {
                let local = self
                    .locals()
                    .get(index as usize)
                    .expect("closed local place has no allocation");
                ops::ClosedPlace {
                    raw: place,
                    kind: ops::ClosedPlaceKind::Local {
                        index,
                        kind: local.kind,
                    },
                    representation: local.representation,
                    rank: u32::try_from(local.extents.len())
                        .expect("closed local rank exceeds u32"),
                    extents: local.extents.clone(),
                    geometry: emission.locals[index as usize].geometry.clone(),
                    words: ops::ClosedPlaceWords::Local(emission.locals[index as usize].words),
                    realization: Some(emission.locals[index as usize].realization),
                }
            }
        }
    }

    pub fn closed_dense_place(
        &self,
        place: ops::PlaceRef,
        emission: &crate::target::KernelEmissionLayout,
    ) -> ops::ClosedDensePlace {
        let closed = self.closed_place(place, emission);
        let geometry = closed.geometry.dense();
        closed.map_geometry(geometry)
    }

    pub fn closed_readable_place(
        &self,
        place: ops::PlaceRef,
        emission: &crate::target::KernelEmissionLayout,
    ) -> ops::ClosedReadablePlace {
        let closed = self.closed_place(place, emission);
        let geometry = closed.geometry.readable();
        closed.map_geometry(geometry)
    }

    fn closed_global_place(
        &self,
        slot: BindingSlot,
        emission: &crate::target::KernelEmissionLayout,
    ) -> ops::ClosedGlobalPlace {
        let binding = &self.interface().bindings[slot.ordinal() as usize];
        let layout = &emission.bindings[slot.ordinal() as usize];
        ops::ClosedGlobalPlace {
            slot,
            access: binding.access,
            buffer_ordinal: slot.ordinal(),
            representation: binding.view.representation,
            rank: binding.rank,
            extents: binding.extents.clone(),
            geometry: layout.geometry.clone(),
            words: layout.words,
        }
    }
    pub fn closed_op<'a>(
        &'a self,
        op: &'a ops::Op<B>,
        emission: &crate::target::KernelEmissionLayout,
    ) -> ops::ClosedOpView<'a, B> {
        use ops::{ClosedBoolValue as Bool, ClosedIndexValue as Index, ClosedOpView as Closed, Op};
        let value = |raw| self.closed_value(raw);
        let index = |raw| {
            let closed = value(raw);
            assert_eq!(
                closed.ty,
                ops::ValueType::Index,
                "closed index operand has non-index type"
            );
            Index(closed)
        };
        let boolean = |raw| {
            let closed = value(raw);
            assert_eq!(
                closed.ty,
                ops::ValueType::Bool,
                "closed boolean operand has non-boolean type"
            );
            Bool(closed)
        };
        let indices = |items: &[ops::ErasedValue]| items.iter().copied().map(index).collect();
        match op {
            Op::Constant {
                out,
                value: constant,
            } => Closed::Constant {
                out: value(*out),
                value: *constant,
            },
            Op::Binary { op, out, a, b } => Closed::Binary {
                op: *op,
                out: value(*out),
                a: value(*a),
                b: value(*b),
            },
            Op::Unary { op, out, a } => Closed::Unary {
                op: *op,
                out: value(*out),
                a: value(*a),
            },
            Op::Bit { op, out, a, b } => Closed::Bit {
                op: *op,
                out: value(*out),
                a: value(*a),
                b: value(*b),
            },
            Op::Fma { out, a, b, c } => Closed::Fma {
                out: value(*out),
                a: value(*a),
                b: value(*b),
                c: value(*c),
            },
            Op::VectorSplat { out, value: scalar } => Closed::VectorSplat {
                out: value(*out),
                value: value(*scalar),
            },
            Op::VectorBinary { op, out, a, b } => Closed::VectorBinary {
                op: *op,
                out: value(*out),
                a: value(*a),
                b: value(*b),
            },
            Op::VectorUnary { op, out, a } => Closed::VectorUnary {
                op: *op,
                out: value(*out),
                a: value(*a),
            },
            Op::VectorBit { op, out, a, b } => Closed::VectorBit {
                op: *op,
                out: value(*out),
                a: value(*a),
                b: value(*b),
            },
            Op::VectorFma { out, a, b, c } => Closed::VectorFma {
                out: value(*out),
                a: value(*a),
                b: value(*b),
                c: value(*c),
            },
            Op::VectorCast { out, a, to } => Closed::VectorCast {
                out: value(*out),
                a: value(*a),
                to: *to,
            },
            Op::VectorLane { out, vector, lane } => Closed::VectorLane {
                out: value(*out),
                vector: value(*vector),
                lane: *lane,
            },
            Op::VectorReduceAdd { out, vector } => Closed::VectorReduceAdd {
                out: value(*out),
                vector: value(*vector),
            },
            Op::Math {
                op,
                precision,
                out,
                a,
            } => {
                assert_eq!(
                    *precision,
                    ops::MathPrecision::Approximate,
                    "exact math must expand before kernel close"
                );
                Closed::ApproximateMath {
                    op: *op,
                    out: value(*out),
                    a: value(*a),
                }
            }
            Op::Cast { out, a, to } => Closed::Cast {
                out: value(*out),
                a: value(*a),
                to: *to,
            },
            Op::Bitcast { out, a, to } => Closed::Bitcast {
                out: value(*out),
                a: value(*a),
                to: *to,
            },
            Op::Cmp { op, out, a, b } => Closed::Cmp {
                op: *op,
                out: boolean(*out),
                a: value(*a),
                b: value(*b),
            },
            Op::Select { out, cond, a, b } => Closed::Select {
                out: value(*out),
                condition: boolean(*cond),
                a: value(*a),
                b: value(*b),
            },
            Op::Logic { op, out, a, b } => Closed::Logic {
                op: *op,
                out: boolean(*out),
                a: boolean(*a),
                b: boolean(*b),
            },
            Op::Not { out, a } => Closed::Not {
                out: boolean(*out),
                a: boolean(*a),
            },
            Op::Geometry { out, kind } => Closed::Geometry {
                out: index(*out),
                kind: *kind,
            },
            Op::NatArg {
                out,
                index: ordinal,
            } => Closed::NatArg {
                out: index(*out),
                index: *ordinal,
                expression: self.interface().nat_args[*ordinal as usize],
            },
            Op::ScalarArg {
                out,
                index: ordinal,
            } => {
                let (symbol, dtype) = self.interface().scalar_args[*ordinal as usize];
                Closed::ScalarArg {
                    out: value(*out),
                    index: *ordinal,
                    symbol,
                    dtype,
                }
            }
            Op::Read {
                out,
                place,
                representation,
                index: raw_indices,
            } => {
                let place = self.closed_place(*place, emission);
                assert_eq!(
                    place.representation, *representation,
                    "closed read representation differs from its place"
                );
                let geometry = place.geometry.readable();
                let place = place.map_geometry(geometry);
                Closed::Read {
                    out: value(*out),
                    place,
                    indices: indices(raw_indices),
                }
            }
            Op::VectorRead {
                out,
                place,
                representation,
                index: raw_indices,
                axis,
                active,
            } => {
                let place = self.closed_place(*place, emission);
                assert_eq!(
                    place.representation, *representation,
                    "closed vector read representation differs from its place"
                );
                assert!(
                    *axis < place.rank,
                    "closed vector read axis is outside rank"
                );
                let geometry = place.geometry.readable();
                let place = place.map_geometry(geometry);
                Closed::VectorRead {
                    out: value(*out),
                    place,
                    indices: indices(raw_indices),
                    axis: *axis,
                    active: index(*active),
                }
            }
            Op::VectorWrite {
                place,
                representation,
                index: raw_indices,
                axis,
                active,
                value: raw_value,
            } => {
                let place = self.closed_place(*place, emission);
                assert_eq!(
                    place.representation, *representation,
                    "closed vector write representation differs from its place"
                );
                assert!(
                    *axis < place.rank,
                    "closed vector write axis is outside rank"
                );
                let value = value(*raw_value);
                let geometry = place.geometry.dense();
                assert_eq!(
                    value.ty,
                    ops::ValueType::Vector {
                        dtype: geometry.dtype,
                        lanes: match value.ty {
                            ops::ValueType::Vector { lanes, .. } => lanes,
                            _ => panic!("closed vector write value is not a vector"),
                        },
                    },
                    "closed vector write element type differs from its place"
                );
                let place = place.map_geometry(geometry);
                Closed::VectorWrite {
                    place,
                    indices: indices(raw_indices),
                    axis: *axis,
                    active: index(*active),
                    value,
                }
            }
            Op::ReadPlane {
                out,
                place,
                plane,
                index: raw_indices,
            } => {
                let place = self.closed_place(*place, emission);
                let geometry = place.geometry.packed();
                let plane_info = geometry.layout.planes[*plane as usize].clone();
                let place = place.map_geometry(geometry);
                Closed::ReadPlane {
                    out: value(*out),
                    place,
                    plane: *plane,
                    plane_info,
                    indices: indices(raw_indices),
                }
            }
            Op::RepresentationConvertPacket {
                source,
                destination,
                conversion,
                packet,
            } => {
                let recipe = seismic_lang::registry::representation_conversion_info(*conversion);
                let source = self.closed_global_place(*source, emission);
                let destination = self.closed_global_place(*destination, emission);
                assert_eq!(
                    source.representation, recipe.source,
                    "closed conversion source differs from recipe"
                );
                assert_eq!(
                    destination.representation, recipe.destination,
                    "closed conversion destination differs from recipe"
                );
                let source_geometry = source.geometry.external();
                let destination_geometry = destination.geometry.packed();
                let source = source.map_geometry(source_geometry);
                let destination = destination.map_geometry(destination_geometry);
                Closed::RepresentationConvertPacket {
                    source,
                    destination,
                    conversion: *conversion,
                    recipe,
                    packet: index(*packet),
                }
            }
            Op::Write {
                place,
                representation,
                index: raw_indices,
                value: raw_value,
            } => {
                let place = self.closed_place(*place, emission);
                assert_eq!(
                    place.representation, *representation,
                    "closed write representation differs from its place"
                );
                let geometry = place.geometry.dense();
                let place = place.map_geometry(geometry);
                Closed::Write {
                    place,
                    indices: indices(raw_indices),
                    value: value(*raw_value),
                }
            }
            Op::Extent { out, place, axis } => Closed::Extent {
                out: index(*out),
                place: self.closed_place(*place, emission),
                axis: *axis,
            },
            Op::Atomic {
                op,
                place,
                representation,
                index: raw_indices,
                value: raw_value,
            } => {
                let place = self.closed_place(*place, emission);
                assert_eq!(
                    place.representation, *representation,
                    "closed atomic representation differs from its place"
                );
                let geometry = place.geometry.dense();
                let place = place.map_geometry(geometry);
                Closed::Atomic {
                    op: *op,
                    place,
                    indices: indices(raw_indices),
                    value: value(*raw_value),
                }
            }
            Op::StoreSlot {
                slot,
                value: raw_value,
                election,
            } => Closed::StoreSlot {
                slot: *slot,
                dtype: self.interface().result_slots[*slot as usize].1,
                value: value(*raw_value),
                election: *election,
            },
            Op::Barrier(scope) => Closed::Barrier(*scope),
            Op::Intrinsic {
                intrinsic,
                op,
                outs,
                args,
                mapping_dependencies,
            } => Closed::Intrinsic {
                intrinsic: *intrinsic,
                signature: seismic_lang::registry::intrinsic_signature(*intrinsic),
                op,
                outputs: outs.iter().copied().map(value).collect(),
                arguments: args.iter().copied().map(value).collect(),
                mapping_dependencies: mapping_dependencies.iter().copied().map(index).collect(),
            },
            Op::Branch {
                cond,
                then,
                otherwise,
                outs,
            } => Closed::Branch {
                condition: boolean(*cond),
                then_block: *then,
                else_block: *otherwise,
                outputs: outs.iter().copied().map(value).collect(),
                then_yields: self
                    .block(*then)
                    .ops
                    .last()
                    .and_then(|op| match op {
                        Op::Yield { values } => Some(values.iter().copied().map(value).collect()),
                        _ => None,
                    })
                    .expect("closed branch then-block ends in Yield"),
                else_yields: self
                    .block(*otherwise)
                    .ops
                    .last()
                    .and_then(|op| match op {
                        Op::Yield { values } => Some(values.iter().copied().map(value).collect()),
                        _ => None,
                    })
                    .expect("closed branch else-block ends in Yield"),
            },
            Op::Repeat {
                start,
                end,
                binder,
                carries_in,
                carry_params,
                body,
                outs,
            } => Closed::Repeat {
                start: index(*start),
                end: index(*end),
                binder: index(*binder),
                carries_in: carries_in.iter().copied().map(value).collect(),
                carry_parameters: carry_params.iter().copied().map(value).collect(),
                body: *body,
                outputs: outs.iter().copied().map(value).collect(),
                body_yields: self
                    .block(*body)
                    .ops
                    .last()
                    .and_then(|op| match op {
                        Op::Yield { values } => Some(values.iter().copied().map(value).collect()),
                        _ => None,
                    })
                    .expect("closed repeat body ends in Yield"),
            },
            Op::Yield { values } => Closed::Yield {
                values: values.iter().copied().map(value).collect(),
            },
        }
    }
    /// Number of typed SSA values in the closed kernel. Codegen resource
    /// models use this closed inventory; native compilers may not reflect a
    /// different legality fact later.
    pub fn value_count(&self) -> u32 {
        u32::try_from(self.inner.value_types().len()).expect("kernel value ordinal space exhausted")
    }
    /// Numerical facts recorded during construction (fma, approximate
    /// math, reassociated intrinsics), consumed by numerics derivation.
    pub fn numerical_facts(&self) -> &[ops::NumericalFact] {
        self.inner.numerical_facts()
    }
    /// Barriers, subgroup use, and local bytes as declared, for resource
    /// derivation.
    pub fn resource_facts(&self) -> &ops::ResourceFacts {
        self.inner.resource_facts()
    }
}

pub(crate) mod internals;
mod reference_math;

/// Canonical identity of the builder-time exact-math recipe set. Changing
/// any recipe changes the digest and invalidates native artifacts.
pub fn reference_math_identity() -> (&'static str, [u8; 32]) {
    (reference_math::VERSION, reference_math::digest())
}
