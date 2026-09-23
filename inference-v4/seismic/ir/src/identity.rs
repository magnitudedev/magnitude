//! Content digests for implementation and variant identity (spec §15.1).
//!
//! Stable identity is encoded explicitly from canonical ordinals and registry
//! identities. Process-local owner/arena/program nonces and Debug renderings
//! never enter these digests.

use seismic_lang::ids::{IntrinsicId, RepresentationId};
use seismic_lang::types::DType;
use sha2::{Digest, Sha256};
use std::fmt;
use std::hash::{Hash, Hasher};
use std::num::NonZeroU64;
use std::sync::atomic::{AtomicU64, Ordering};

/// Process-local construction identity. Numeric ordinals are meaningful only
/// inside the owner which minted them; putting the owner in every construction
/// handle makes cross-implementation mixing observable at the first builder
/// operation instead of at a later table lookup.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) struct OwnerToken(NonZeroU64);

impl OwnerToken {
    pub(crate) fn fresh() -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(1);
        let value = NEXT
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |value| {
                value.checked_add(1)
            })
            .unwrap_or_else(|_| panic!("implementation owner identity space exhausted"));
        let value = NonZeroU64::new(value).expect("owner counter starts nonzero");
        Self(value)
    }
}

impl fmt::Debug for OwnerToken {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Owner identity is process-local validity metadata, not semantic
        // structure. Keeping it out of Debug also keeps derived structural
        // renderings deterministic across processes.
        f.write_str("owner")
    }
}

impl Hash for OwnerToken {
    fn hash<H: Hasher>(&self, _state: &mut H) {
        // See Debug: owner-branded handles hash by their canonical ordinal
        // and structure. Owner equality is still enforced before such a
        // handle enters a closed artifact or structural digest.
    }
}

pub struct StructureDigest {
    sha: Sha256,
}

/// Sealed-format encoder used by a backend to identify the payload of one
/// typed intrinsic op. Backends cannot inject opaque bytes or a Debug string;
/// they exhaustively write a variant tag followed by typed fields.
pub struct IntrinsicIdentityBuilder {
    digest: StructureDigest,
}

impl IntrinsicIdentityBuilder {
    pub fn new() -> Self {
        Self {
            digest: StructureDigest::new("seismic-intrinsic-identity-v1"),
        }
    }
    pub fn variant(&mut self, name: &'static str) {
        self.digest.bytes(name.as_bytes());
    }
    pub fn bool(&mut self, value: bool) {
        self.digest.bool(value);
    }
    pub fn u32(&mut self, value: u32) {
        self.digest.u32(value);
    }
    pub fn u64(&mut self, value: u64) {
        self.digest.u64(value);
    }
    pub fn i64(&mut self, value: i64) {
        self.digest.i64(value);
    }
    pub fn dtype(&mut self, value: DType) {
        self.digest.bytes(value.name().as_bytes());
    }
    pub fn representation(&mut self, value: RepresentationId) {
        self.digest.bytes(
            seismic_lang::registry::representation_info(value)
                .name
                .as_bytes(),
        );
    }
    pub fn intrinsic(&mut self, value: IntrinsicId) {
        let signature = seismic_lang::registry::intrinsic_signature(value);
        let capability = seismic_lang::registry::capability_info(signature.capability);
        self.digest.bytes(capability.backend.as_str().as_bytes());
        self.digest.bytes(capability.name.as_bytes());
        self.digest.bytes(signature.name.as_bytes());
    }
    pub fn place(&mut self, value: crate::kernel::ops::PlaceRef) {
        match value {
            crate::kernel::ops::PlaceRef::Global { slot } => {
                self.variant("global");
                self.u32(slot.ordinal());
            }
            crate::kernel::ops::PlaceRef::Local { index } => {
                self.variant("local");
                self.u32(index);
            }
        }
    }
    pub fn logical_tensor(&mut self, tensor: &crate::kernel::ops::LogicalTensorMap) {
        use crate::kernel::ops::{LogicalSliceAxis, LogicalViewStep};
        self.place(tensor.base);
        self.representation(tensor.representation);
        self.digest.len(tensor.extents.len());
        for extent in &tensor.extents {
            self.u32(extent.ordinal());
        }
        self.digest.len(tensor.steps.len());
        for step in &tensor.steps {
            match step {
                LogicalViewStep::Plane { plane, axis } => {
                    self.variant("plane");
                    self.u32(*plane);
                    self.u32(*axis);
                }
                LogicalViewStep::Slice(axes) => {
                    self.variant("slice");
                    self.digest.len(axes.len());
                    for axis in axes {
                        match axis {
                            LogicalSliceAxis::Point(value) => {
                                self.variant("point");
                                self.u32(value.ordinal());
                            }
                            LogicalSliceAxis::Range { start, end } => {
                                self.variant("range");
                                self.u32(start.ordinal());
                                self.u32(end.ordinal());
                            }
                            LogicalSliceAxis::Full => self.variant("full"),
                        }
                    }
                }
                LogicalViewStep::Transpose(permutation) => {
                    self.variant("transpose");
                    self.digest.len(permutation.len());
                    for axis in permutation {
                        self.u32(*axis);
                    }
                }
                LogicalViewStep::Reshape { from, to } => {
                    self.variant("reshape");
                    self.digest.len(from.len());
                    for extent in from {
                        self.u32(extent.ordinal());
                    }
                    self.digest.len(to.len());
                    for extent in to {
                        self.u32(extent.ordinal());
                    }
                }
            }
        }
    }
    pub fn finish(self) -> [u8; 32] {
        self.digest.finish()
    }
}

impl StructureDigest {
    pub fn new(domain: &'static str) -> Self {
        let mut sha = Sha256::new();
        sha.update(domain.as_bytes());
        sha.update([0u8]);
        Self { sha }
    }

    pub fn bytes(&mut self, bytes: &[u8]) {
        self.sha.update((bytes.len() as u64).to_le_bytes());
        self.sha.update(bytes);
    }

    pub fn bool(&mut self, value: bool) {
        self.sha.update([u8::from(value)]);
    }
    pub fn u32(&mut self, value: u32) {
        self.sha.update(value.to_le_bytes());
    }
    pub fn u64(&mut self, value: u64) {
        self.sha.update(value.to_le_bytes());
    }
    pub fn i64(&mut self, value: i64) {
        self.sha.update(value.to_le_bytes());
    }
    pub fn len(&mut self, value: usize) {
        self.u64(u64::try_from(value).expect("structural length exceeds u64"));
    }

    pub fn hashed<T: CanonicalIdentity + ?Sized>(&mut self, value: &T) {
        value.encode_identity(self);
    }

    pub fn finish(self) -> [u8; 32] {
        let output = self.sha.finalize();
        let mut digest = [0u8; 32];
        digest.copy_from_slice(&output);
        digest
    }
}

pub trait CanonicalIdentity {
    fn encode_identity(&self, out: &mut StructureDigest);
}

macro_rules! fixed {
    ($ty:ty, $method:ident) => {
        impl CanonicalIdentity for $ty {
            fn encode_identity(&self, out: &mut StructureDigest) {
                out.$method(*self);
            }
        }
    };
}
fixed!(u32, u32);
fixed!(u64, u64);
fixed!(i64, i64);
fixed!(bool, bool);
impl CanonicalIdentity for u8 {
    fn encode_identity(&self, out: &mut StructureDigest) {
        out.sha.update([*self]);
    }
}
impl CanonicalIdentity for u16 {
    fn encode_identity(&self, out: &mut StructureDigest) {
        out.sha.update(self.to_le_bytes());
    }
}
impl CanonicalIdentity for i32 {
    fn encode_identity(&self, out: &mut StructureDigest) {
        out.sha.update(self.to_le_bytes());
    }
}
impl CanonicalIdentity for usize {
    fn encode_identity(&self, out: &mut StructureDigest) {
        out.len(*self);
    }
}
impl<T: CanonicalIdentity> CanonicalIdentity for Option<T> {
    fn encode_identity(&self, out: &mut StructureDigest) {
        match self {
            Some(v) => {
                out.bool(true);
                v.encode_identity(out);
            }
            None => out.bool(false),
        }
    }
}
impl<T: CanonicalIdentity> CanonicalIdentity for [T] {
    fn encode_identity(&self, out: &mut StructureDigest) {
        out.len(self.len());
        for v in self {
            v.encode_identity(out);
        }
    }
}
impl<T: CanonicalIdentity> CanonicalIdentity for Vec<T> {
    fn encode_identity(&self, out: &mut StructureDigest) {
        self.as_slice().encode_identity(out);
    }
}
impl<A: CanonicalIdentity, C: CanonicalIdentity> CanonicalIdentity for (A, C) {
    fn encode_identity(&self, out: &mut StructureDigest) {
        self.0.encode_identity(out);
        self.1.encode_identity(out);
    }
}
impl<A: CanonicalIdentity, C: CanonicalIdentity, D: CanonicalIdentity> CanonicalIdentity
    for (A, C, D)
{
    fn encode_identity(&self, out: &mut StructureDigest) {
        self.0.encode_identity(out);
        self.1.encode_identity(out);
        self.2.encode_identity(out);
    }
}

impl CanonicalIdentity for DType {
    fn encode_identity(&self, out: &mut StructureDigest) {
        out.bytes(self.name().as_bytes());
    }
}
impl CanonicalIdentity for crate::repr::ScalarKind {
    fn encode_identity(&self, out: &mut StructureDigest) {
        match self {
            Self::Scalar(dtype) => {
                out.bytes(b"source-scalar");
                out.hashed(dtype);
            }
            Self::Nat64 => out.bytes(b"natural-u64"),
        }
    }
}
impl CanonicalIdentity for seismic_lang::registry::BackendName {
    fn encode_identity(&self, out: &mut StructureDigest) {
        out.bytes(self.as_str().as_bytes());
    }
}
impl CanonicalIdentity for seismic_lang::expr::SymbolSort {
    fn encode_identity(&self, out: &mut StructureDigest) {
        use seismic_lang::expr::SymbolSort::*;
        match self {
            Nat => out.bytes(b"nat"),
            Int => out.bytes(b"int"),
            Scalar(dtype) => {
                out.bytes(b"scalar");
                dtype.encode_identity(out);
            }
        }
    }
}
impl CanonicalIdentity for crate::kernel::ops::BindingAccess {
    fn encode_identity(&self, out: &mut StructureDigest) {
        out.bytes(match self {
            Self::Read => b"read",
            Self::Write => b"write",
        });
    }
}
impl CanonicalIdentity for crate::storage::LaunchLocalKind {
    fn encode_identity(&self, out: &mut StructureDigest) {
        out.bytes(match self {
            Self::Workgroup => b"workgroup",
            Self::Participant => b"participant",
            Self::Register => b"register",
        });
    }
}

macro_rules! tagged_enum {
    ($ty:path, {$($variant:ident => $tag:literal),+ $(,)?}) => { impl CanonicalIdentity for $ty { fn encode_identity(&self, out: &mut StructureDigest) { out.bytes(match self { $(Self::$variant => $tag),+ }); } } };
}
tagged_enum!(crate::kernel::ops::BinaryOp, {Add=>b"add",Sub=>b"sub",Mul=>b"mul",Div=>b"div",Rem=>b"rem",Min=>b"min",Max=>b"max"});
tagged_enum!(crate::kernel::ops::UnaryOp, {Neg=>b"neg",Abs=>b"abs"});
tagged_enum!(crate::kernel::ops::BitOp, {And=>b"and",Or=>b"or",Xor=>b"xor",Shl=>b"shl",Shr=>b"shr"});
tagged_enum!(crate::kernel::ops::CmpOp, {Eq=>b"eq",Ne=>b"ne",Lt=>b"lt",Le=>b"le",Gt=>b"gt",Ge=>b"ge"});
tagged_enum!(crate::kernel::ops::LogicOp, {And=>b"and",Or=>b"or"});
tagged_enum!(crate::kernel::ops::MathPrecision, {Exact=>b"exact",Approximate=>b"approximate"});
tagged_enum!(crate::kernel::ops::BarrierScope, {Workgroup=>b"workgroup",Subgroup=>b"subgroup"});
tagged_enum!(crate::kernel::ops::StoreElection, {GlobalLeader=>b"global-leader"});
tagged_enum!(crate::schedule::LaunchParticipation, {Independent=>b"independent",CooperativeGrid=>b"cooperative-grid"});
tagged_enum!(seismic_lang::intrinsics::AtomicOp, {Add=>b"add",Max=>b"max",Min=>b"min"});
tagged_enum!(seismic_lang::intrinsics::MathOp, {Exp=>b"exp",Fma=>b"fma",Rsqrt=>b"rsqrt",Sqrt=>b"sqrt",Log=>b"log",Sin=>b"sin",Cos=>b"cos",Abs=>b"abs",Max=>b"max",Min=>b"min"});
tagged_enum!(crate::target::ResourceOwnershipScope, {Participant=>b"participant",Subgroup=>b"subgroup",Workgroup=>b"workgroup"});
tagged_enum!(crate::target::AddressableResourceRealization, {Native=>b"native"});
tagged_enum!(crate::target::ResourceLifetime, {Operation=>b"operation",Segment=>b"segment",Launch=>b"launch"});

impl CanonicalIdentity for crate::kernel::ops::GeometryValue {
    fn encode_identity(&self, out: &mut StructureDigest) {
        use crate::kernel::ops::GeometryValue::*;
        match self {
            WorkgroupId(a) => {
                out.bytes(b"workgroup-id");
                a.encode_identity(out)
            }
            LocalId(a) => {
                out.bytes(b"local-id");
                a.encode_identity(out)
            }
            GlobalId(a) => {
                out.bytes(b"global-id");
                a.encode_identity(out)
            }
            WorkgroupSize(a) => {
                out.bytes(b"workgroup-size");
                a.encode_identity(out)
            }
            GridSize(a) => {
                out.bytes(b"grid-size");
                a.encode_identity(out)
            }
            SubgroupLane => out.bytes(b"subgroup-lane"),
            SubgroupOrdinal => out.bytes(b"subgroup-ordinal"),
            SubgroupSize => out.bytes(b"subgroup-size"),
        }
    }
}
impl CanonicalIdentity for crate::kernel::ops::ValueType {
    fn encode_identity(&self, out: &mut StructureDigest) {
        use crate::kernel::ops::ValueType::*;
        match self {
            Scalar(dtype) => {
                out.bytes(b"scalar");
                dtype.encode_identity(out)
            }
            Vector { dtype, lanes } => {
                out.bytes(b"vector");
                dtype.encode_identity(out);
                lanes.encode_identity(out)
            }
            Index => out.bytes(b"index"),
            Bool => out.bytes(b"bool"),
            Opaque { name } => {
                out.bytes(b"opaque");
                out.bytes(name.as_bytes())
            }
        }
    }
}
impl CanonicalIdentity for crate::storage::ScheduleRegionEdge {
    fn encode_identity(&self, out: &mut StructureDigest) {
        use crate::storage::ScheduleRegionEdge::*;
        match self {
            IfThen {
                node,
                parent_ordinal,
            } => {
                out.bytes(b"if-then");
                node.encode_identity(out);
                parent_ordinal.encode_identity(out)
            }
            IfElse {
                node,
                parent_ordinal,
            } => {
                out.bytes(b"if-else");
                node.encode_identity(out);
                parent_ordinal.encode_identity(out)
            }
            RepeatBody {
                node,
                parent_ordinal,
            } => {
                out.bytes(b"repeat");
                node.encode_identity(out);
                parent_ordinal.encode_identity(out)
            }
            Imported {
                node,
                parent_ordinal,
            } => {
                out.bytes(b"imported");
                node.encode_identity(out);
                parent_ordinal.encode_identity(out)
            }
        }
    }
}
impl CanonicalIdentity for crate::schedule::FillValue {
    fn encode_identity(&self, out: &mut StructureDigest) {
        out.bytes(match self {
            Self::U8(_) => b"u8",
            Self::U16(_) => b"u16",
            Self::U32(_) => b"u32",
        });
        out.bytes(self.pattern());
    }
}
impl CanonicalIdentity for crate::kernel::ops::ResourceFacts {
    fn encode_identity(&self, out: &mut StructureDigest) {
        self.local_kinds.encode_identity(out);
        self.barriers.encode_identity(out);
        self.uses_subgroup.encode_identity(out);
        self.binding_count.encode_identity(out);
    }
}

impl CanonicalIdentity for () {
    fn encode_identity(&self, out: &mut StructureDigest) {
        out.bytes(b"unit");
    }
}

impl CanonicalIdentity for seismic_lang::failure::SourceFailure {
    fn encode_identity(&self, out: &mut StructureDigest) {
        use seismic_lang::failure::SourceFailureCause;
        use seismic_lang::reference_math::ScalarFailure;
        use seismic_lang::entry::CheckReason;
        out.bytes(b"source-failure");
        out.bytes(self.event.body().digest());
        for ordinal in self.event.position() { out.u32(ordinal); }
        match &self.cause {
            SourceFailureCause::Scalar(ScalarFailure::IntegerDivisionByZero) => out.bytes(b"division-zero"),
            SourceFailureCause::Scalar(ScalarFailure::SignedDivisionOverflow) => out.bytes(b"division-overflow"),
            SourceFailureCause::Scalar(ScalarFailure::ShiftCount) => out.bytes(b"shift-count"),
            SourceFailureCause::Check(CheckReason::IndexBound) => out.bytes(b"index-bound"),
            SourceFailureCause::Check(CheckReason::RangeOrder) => out.bytes(b"range-order"),
            SourceFailureCause::Check(CheckReason::Custom(message)) => { out.bytes(b"custom-check"); out.bytes(message.as_bytes()); }
        }
    }
}
