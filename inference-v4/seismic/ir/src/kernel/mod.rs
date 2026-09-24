//! Kernel IR (spec §7).
//!
//! Kernels are built only by the source-directed `PortableBuilder`
//! (`internals`). Lexical construction: a value created in a block is usable
//! only in that block or a dominated child; branches own their join and
//! repeats own their carries, and both close only with identical result
//! schemas. Materialization is a compiler operation performed by the
//! implementation builder, never by source.
//!
//! The IR is consumed by native compilers through the read surface of
//! [`Kernel`].

use crate::identity::OwnerToken;
use crate::target::PhysicalDialect;

pub mod ops;
mod representation;

// ---------------------------------------------------------------------------
// Handles
// ---------------------------------------------------------------------------

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
    pub fn index(self) -> u32 {
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
    pub fn index(self) -> u32 {
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
    pub fn index(self) -> u32 {
        self.ordinal()
    }
}

// ---------------------------------------------------------------------------
// Read surface for native compilers
// ---------------------------------------------------------------------------

/// The arena of every kernel of one implementation (or one frozen plan).
#[derive(Debug)]
pub struct KernelArena<B: PhysicalDialect> {
    inner: internals::Arena<B>,
}

impl<B: PhysicalDialect> KernelArena<B> {
    pub fn retained_bytes(&self) -> usize {
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
pub struct Kernel<B: PhysicalDialect> {
    inner: internals::KernelData<B>,
}

impl<B: PhysicalDialect> Kernel<B> {
    pub fn blocks(&self) -> &[ops::Block<B>] {
        self.inner.blocks()
    }
    pub fn block_multiplicity(&self) -> &[Option<seismic_lang::expr::NatExpr>] {
        self.inner.block_multiplicity()
    }
    pub fn value_types(&self) -> impl ExactSizeIterator<Item = &ops::ValueType> {
        self.inner.value_types()
    }
    /// Exact host expression derived for this SSA value during construction.
    /// A device-produced value can have no such expression; consumers must not
    /// replace its actual logical extent with backing capacity.
    pub fn exact_nat(&self, value: ops::ErasedValue) -> Option<seismic_lang::expr::NatExpr> {
        self.inner.exact_nat(value)
    }
    pub fn intrinsic_resources(&self) -> &[ops::IntrinsicResources] {
        self.inner.intrinsic_resources()
    }
    pub fn intrinsics_used(&self) -> &[seismic_lang::ids::IntrinsicId] {
        self.inner.intrinsics_used()
    }
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
            Op::VectorFromLanes { out, lanes } => Closed::VectorFromLanes {
                out: value(*out),
                lanes: lanes.iter().map(|lane| value(*lane)).collect(),
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
            Op::ScalarBits { out, a } => Closed::ScalarBits {
                out: value(*out),
                a: value(*a),
            },
            Op::ScalarFromBits { out, a } => Closed::ScalarFromBits {
                out: value(*out),
                a: value(*a),
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
                let (symbol, kind) = self.interface().scalar_args[*ordinal as usize];
                Closed::ScalarArg {
                    out: value(*out),
                    index: *ordinal,
                    symbol,
                    kind,
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
                let geometry = place.geometry.dense();
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
                let geometry = place.geometry.dense();
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
            Op::ReadPlaneField {
                out,
                place,
                plane,
                field,
                index: raw_indices,
            } => {
                let place = self.closed_place(*place, emission);
                let geometry = place.geometry.packed();
                let plane_info = geometry.layout.planes[*plane as usize].clone();
                let place = place.map_geometry(geometry);
                Closed::ReadPlaneField {
                    out: value(*out),
                    place,
                    plane: *plane,
                    field: *field,
                    plane_info,
                    indices: indices(raw_indices),
                }
            }
            Op::ReadPlane {
                out,
                place,
                plane,
                element,
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
                    element: index(*element),
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
                kind: self.interface().result_slots[*slot as usize].kind(),
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

/// Checked construction for semantics whose value types are known at runtime.
pub mod dynamic {
    pub use super::internals::{
        LogicalIndexBinding, PortableBranch, PortableBuilder, PortableCursor, PortablePlace,
        PortableRepeat, PortableSliceAxis, PortableTensor, PortableValue,
    };
}
