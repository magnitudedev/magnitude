//! Typed registry identities (spec §3.3, §4.2, §7.1).
//!
//! Capabilities, intrinsic signatures, and representations are interned
//! typed ids, never strings. The registry is static and sealed: every id is
//! valid for the lifetime of the process, and lookups by name exist only at
//! the source-checking and target-profiling boundaries.
//!
//! W1 owns the interning over the existing `intrinsics` and `repr` tables.

use crate::ids::{CapabilityId, IntrinsicId, RepresentationConversionId, RepresentationId};
use crate::types::DType;

pub use crate::repr::{
    CodeInterpretation, DecodeRecipe, DecodeStep, DecodeTemp, FloatCodeFormat, PlaneEncoding,
    PlaneField, PlaneSchema,
};

/// Backends are a closed set.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum BackendName {
    Cpu,
    Metal,
    Cuda,
}

impl BackendName {
    pub const ALL: [BackendName; 3] = [BackendName::Cpu, BackendName::Metal, BackendName::Cuda];

    pub fn parse(name: &str) -> Option<BackendName> {
        match name {
            "cpu" => Some(BackendName::Cpu),
            "metal" => Some(BackendName::Metal),
            "cuda" => Some(BackendName::Cuda),
            _ => None,
        }
    }

    pub const fn as_str(self) -> &'static str {
        match self {
            BackendName::Cpu => "cpu",
            BackendName::Metal => "metal",
            BackendName::Cuda => "cuda",
        }
    }

    /// Whether source may declare a direct `native … for` implementation on
    /// this backend (the explicitly selected direct-native route, D6).
    pub fn supports_direct_native(self) -> bool {
        match self {
            BackendName::Metal => true,
            BackendName::Cpu | BackendName::Cuda => false,
        }
    }
}

/// Revision of the whole registry. Any semantic change to a primitive,
/// capability, intrinsic, or representation changes this string, and with it
/// every cache identity.
pub const REGISTRY_REVISION: &str = "seismic-registry-v14";

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CapabilityInfo {
    pub id: CapabilityId,
    pub backend: BackendName,
    /// Namespace name within the backend (`matrix`, `subgroup`).
    pub name: &'static str,
}

/// Exact typed signature of one intrinsic, as the kernel builder accepts it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct IntrinsicSignature {
    pub id: IntrinsicId,
    pub capability: CapabilityId,
    pub name: &'static str,
    pub arguments: Vec<IntrinsicArgument>,
    pub result: IntrinsicResultType,
    pub execution: IntrinsicExecution,
    pub effects: IntrinsicEffects,
    pub numerical: IntrinsicNumerics,
}

/// The logical iteration domain in which an intrinsic executes. Physical
/// workgroup/tile geometry remains a compiler decision.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum IntrinsicExecution {
    /// Executes inside a source-authored `parallel for` domain. Lane and
    /// subgroup identities have meaning only in that enclosing domain.
    WithinEnclosingParallel,
    /// Defines a whole-tensor collective. `result` selects the tensor result
    /// whose axes are the logical iteration domain.
    WholeTensor { result: u8 },
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct IntrinsicArgument {
    pub name: &'static str,
    pub category: OperandCategory,
}

/// Operand categories the kernel builder distinguishes (§7.1).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum OperandCategory {
    Scalar(DType),
    /// A readable tensor place of a given representation and rank.
    Readable {
        representation: RepresentationId,
        rank: u32,
    },
    /// A writable tensor place.
    Writable {
        representation: RepresentationId,
        rank: u32,
    },
    /// A backend-opaque value produced by another intrinsic of the same
    /// capability.
    Opaque {
        capability: CapabilityId,
        name: &'static str,
    },
    /// A compile-time constant.
    Constant(DType),
}

impl OperandCategory {
    fn admits(&self, operand: &OperandElement) -> bool {
        match (self, operand) {
            (OperandCategory::Scalar(dtype), OperandElement::Scalar(actual)) => dtype == actual,
            (
                OperandCategory::Readable {
                    representation,
                    rank,
                },
                OperandElement::Tensor {
                    representation: actual,
                    rank: actual_rank,
                },
            ) => representation == actual && rank == actual_rank,
            (OperandCategory::Scalar(_), OperandElement::Tensor { .. })
            | (OperandCategory::Readable { .. }, OperandElement::Scalar(_)) => false,
            (
                OperandCategory::Writable { .. }
                | OperandCategory::Opaque { .. }
                | OperandCategory::Constant(_),
                _,
            ) => unreachable!("intrinsic rows declare only scalar and readable operands"),
        }
    }
}

/// One result axis projected from an actual tensor argument. The ordered
/// projections define result rank as well as geometry.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct IntrinsicResultAxis {
    pub argument: u32,
    pub axis: u32,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum IntrinsicResultType {
    Void,
    Scalar(DType),
    Owned {
        representation: RepresentationId,
        axes: &'static [IntrinsicResultAxis],
    },
    Opaque {
        capability: CapabilityId,
        name: &'static str,
    },
}

#[derive(Clone, Debug, PartialEq, Eq, Default)]
pub struct IntrinsicEffects {
    /// Indices of `Writable` arguments the intrinsic writes.
    pub writes: Vec<u32>,
    /// Which execution cohort must participate together. This owns tail
    /// legality; backends may not infer masked-lane semantics for a full
    /// subgroup/workgroup intrinsic.
    pub participation: IntrinsicParticipation,
    /// Strongest execution scope at which every result is guaranteed equal.
    /// This is a semantic property of the intrinsic, not an emitter guess.
    pub result_uniformity: IntrinsicUniformity,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum IntrinsicParticipation {
    #[default]
    Independent,
    FullSubgroup,
    FullWorkgroup,
    /// A full workgroup of exactly this many participants. A placement fact
    /// like `FullWorkgroup`; the size is consumed by physical construction.
    FixedWorkgroup(u32),
}

/// Language meaning of one intrinsic row.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum IntrinsicDenotation {
    /// Lane index within the cohort of the enclosing parallel loop.
    ParticipantIndex,
    /// Value of the named cohort member.
    Exchange,
    /// Fold over the cohort; association unspecified for `Sum`, exact for
    /// `Max`/`Min`.
    CohortFold { op: crate::intrinsics::ReduceOp },
    /// `out[i, j] = (acc[i, j] +) sum_k a[i, k] * b[k, j]`, association
    /// unspecified.
    MatrixProduct { accumulate: bool },
}

/// The element category of one actual intrinsic operand, as overload
/// resolution compares it against a row's `OperandCategory`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum OperandElement {
    Scalar(DType),
    Tensor {
        representation: RepresentationId,
        rank: u32,
    },
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum IntrinsicUniformity {
    Workgroup,
    Subgroup,
    #[default]
    Varying,
}

/// Exact numerical contract of one intrinsic (§9.2). `Unknown` intrinsics
/// are selectable only under an unconstrained policy.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum IntrinsicNumerics {
    Exact,
    /// Association topology of a reduction or matrix accumulation.
    Reassociated {
        accumulator: DType,
    },
    Approximate {
        ulps: u32,
    },
    Unknown,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RepresentationInfo {
    pub id: RepresentationId,
    pub name: &'static str,
    pub kind: RepresentationKind,
    /// Dtype produced by reading one element.
    pub decoded: DType,
    /// Whether the registry defines canonical logical writes/updates.
    pub access: RepresentationAccess,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum RepresentationAccess {
    ReadWrite,
    ReadOnly,
    ConversionSource,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RepresentationKind {
    Dense(DType),
    Packed(PackedPacketLayout),
    /// Canonical bytes supplied by an external artifact. External packets are
    /// never element-readable or writable; a registered exact conversion is
    /// the sole transition into resident storage.
    External(ExternalPacketLayout),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ExternalPacketLayout {
    pub packet_size: u32,
    pub packet_alignment: u32,
    pub logical_group: u32,
    pub packing_axis: PackingAxis,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RepresentationConversion {
    pub id: RepresentationConversionId,
    pub source: RepresentationId,
    pub destination: RepresentationId,
    pub recipe: PacketRepackRecipe,
}

/// Exact, validated ownership for one external-packet -> resident-packet
/// conversion. There is exactly one recipe for every destination plane and
/// each recipe completely initializes that plane.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PacketRepackRecipe {
    pub planes: Vec<PlaneRepackRecipe>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PlaneRepackRecipe {
    /// One source packet bit ordinal for every destination plane bit.
    BitRoutes(Vec<u32>),
    /// One typed expression for every dense destination plane element.
    DenseValues(Vec<RepackExpr>),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RepackExpr {
    SourceBits { bit: u32, width: u8 },
    ShiftLeft { value: Box<RepackExpr>, bits: u8 },
    BitOr(Box<RepackExpr>, Box<RepackExpr>),
    OffsetI32 { value: Box<RepackExpr>, offset: i32 },
    F16ToF32(Box<RepackExpr>),
    I32ToF32(Box<RepackExpr>),
    MultiplyF32(Box<RepackExpr>, Box<RepackExpr>),
}

/// The one canonical ABI layout of a packed logical packet. Every backend,
/// planner and runtime consumes this descriptor; none reconstructs packed
/// geometry from representation names or decoder implementation details.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PackedPacketLayout {
    /// Physical planes in ABI order, with offsets within one packet.
    pub planes: Vec<PlaneInfo>,
    /// Logical elements represented by one packet along the final logical
    /// axis. Packed representations always pack that axis.
    pub group: u32,
    pub packing_axis: PackingAxis,
    pub packet_size: u32,
    pub packet_alignment: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum PackingAxis {
    Last,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PlaneInfo {
    pub name: &'static str,
    pub offset: u32,
    pub bytes_per_group: u32,
    pub alignment: u32,
    pub encoding: PlaneEncoding,
    pub group: u32,
    pub fields: u32,
    pub entry_bits: u32,
    pub storage_dtype: DType,
}

impl PlaneInfo {
    /// Storage coordinates differ from logical decoded-value coordinates:
    /// integer codes expose words, float codes bytes, coefficients typed elements.
    pub fn storage_element_bytes(&self) -> u32 {
        match self.encoding {
            PlaneEncoding::Dense(dtype) => dtype.bytes(),
            PlaneEncoding::Packed { .. } => 4,
            PlaneEncoding::FloatCode { .. } => 1,
        }
    }
    pub fn storage_elements_per_packet(&self) -> u32 {
        self.bytes_per_group.div_ceil(self.storage_element_bytes())
    }
}

impl PackedPacketLayout {
    /// Packet count for one logical extent on the packing (last) axis.
    pub fn packet_extent(&self, logical_extent: u64) -> u64 {
        logical_extent.div_ceil(u64::from(self.group))
    }

    /// Canonical byte stride between adjacent packed-axis rows.
    pub fn row_bytes(&self, logical_extent: u64) -> Option<u64> {
        self.packet_extent(logical_extent)
            .checked_mul(u64::from(self.packet_size))
    }

    /// Byte offset of a plane within packet `packet`.
    pub fn plane_offset(&self, packet: u64, plane: usize) -> Option<u64> {
        packet
            .checked_mul(u64::from(self.packet_size))?
            .checked_add(u64::from(self.planes.get(plane)?.offset))
    }

    /// Total bytes for `outer_rows` rows of `logical_extent` elements.
    pub fn bytes(&self, outer_rows: u64, logical_extent: u64) -> Option<u64> {
        outer_rows.checked_mul(self.row_bytes(logical_extent)?)
    }

    /// Canonical packets of `rows` rows of `logical_extent` elements, assembled
    /// from plane-separated bytes. `planes` names every plane of this layout
    /// exactly once, in any order. Plane `p` supplies
    /// `rows × packet_extent(logical_extent) × p.bytes_per_group` bytes ordered
    /// by (row, packet); each packet's slice is exactly the bytes that plane
    /// occupies inside one canonical packet. Packet bytes no plane occupies
    /// are zero.
    ///
    /// The caller has established that `bytes(rows, logical_extent)` exists
    /// (the canonical byte count of the tensor fits `u64`); geometry that
    /// overflows is a caller contradiction and panics.
    pub fn packets_from_planes(
        &self,
        rows: u64,
        logical_extent: u64,
        planes: &[(&str, &[u8])],
    ) -> Result<Vec<u8>, PlaneAssemblyError> {
        let total = self
            .bytes(rows, logical_extent)
            .expect("packed tensor geometry has a canonical byte count");
        let packets = rows
            .checked_mul(self.packet_extent(logical_extent))
            .expect("packet count of a tensor with a canonical byte count fits u64");
        if planes.len() != self.planes.len()
            || !self
                .planes
                .iter()
                .all(|plane| planes.iter().filter(|(name, _)| *name == plane.name).count() == 1)
        {
            return Err(PlaneAssemblyError::PlaneSet);
        }
        let supplied: Vec<&[u8]> = self
            .planes
            .iter()
            .map(|plane| {
                let bytes = planes
                    .iter()
                    .find(|(name, _)| *name == plane.name)
                    .expect("every layout plane is supplied exactly once")
                    .1;
                let expected = packets
                    .checked_mul(u64::from(plane.bytes_per_group))
                    .expect("plane bytes of a tensor with a canonical byte count fit u64");
                let actual = bytes.len() as u64;
                if actual == expected {
                    Ok(bytes)
                } else {
                    Err(PlaneAssemblyError::PlaneByteLength {
                        plane: plane.name,
                        expected,
                        actual,
                    })
                }
            })
            .collect::<Result<_, _>>()?;
        let mut canonical = vec![
            0u8;
            usize::try_from(total).expect("canonical packet bytes fit the host address space")
        ];
        let packet_size = self.packet_size as usize;
        for (plane, bytes) in self.planes.iter().zip(supplied) {
            let width = plane.bytes_per_group as usize;
            let offset = plane.offset as usize;
            for (packet, source) in canonical
                .chunks_exact_mut(packet_size)
                .zip(bytes.chunks_exact(width))
            {
                packet[offset..offset + width].copy_from_slice(source);
            }
        }
        Ok(canonical)
    }
}

/// Plane-separated bytes that do not form a packed layout's planes
/// (`PackedPacketLayout::packets_from_planes`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PlaneAssemblyError {
    /// The supplied plane names are not exactly the layout's plane names.
    PlaneSet,
    /// A plane's byte length differs from its canonical length.
    PlaneByteLength {
        plane: &'static str,
        expected: u64,
        actual: u64,
    },
}

pub fn capability(backend: BackendName, name: &str) -> Option<CapabilityId> {
    internals::capability(backend, name)
}
pub fn capability_info(id: CapabilityId) -> &'static CapabilityInfo {
    internals::capability_info(id)
}
pub fn capabilities(backend: BackendName) -> &'static [CapabilityInfo] {
    internals::capabilities(backend)
}
/// Every row of `capability` named `name`, in declaration order. Empty when
/// the capability has no intrinsic of that name.
pub fn intrinsic_overloads(capability: CapabilityId, name: &str) -> &'static [IntrinsicId] {
    internals::intrinsic_overloads(capability, name)
}
/// The row of `overload` whose declared arguments are exactly `operands`.
/// Rows of one overload have pairwise distinct argument lists (the table is
/// built that way), so at most one row matches.
pub fn resolve_intrinsic(
    overload: &[IntrinsicId],
    operands: &[OperandElement],
) -> Option<IntrinsicId> {
    overload.iter().copied().find(|id| {
        let arguments = &intrinsic_signature(*id).arguments;
        arguments.len() == operands.len()
            && arguments
                .iter()
                .zip(operands)
                .all(|(argument, operand)| argument.category.admits(operand))
    })
}
/// The language meaning of one intrinsic row.
pub fn intrinsic_denotation(id: IntrinsicId) -> IntrinsicDenotation {
    internals::intrinsic_denotation(id)
}
pub fn intrinsic_signature(id: IntrinsicId) -> &'static IntrinsicSignature {
    internals::intrinsic_signature(id)
}
pub fn intrinsics(capability: CapabilityId) -> &'static [IntrinsicSignature] {
    internals::intrinsics(capability)
}
pub fn representation(name: &str) -> Option<RepresentationId> {
    internals::representation(name)
}
pub fn representation_info(id: RepresentationId) -> &'static RepresentationInfo {
    internals::representation_info(id)
}
pub fn representations() -> &'static [RepresentationInfo] {
    internals::representations()
}
pub fn representation_conversion(
    source: RepresentationId,
    destination: RepresentationId,
) -> Option<&'static RepresentationConversion> {
    internals::representation_conversion(source, destination)
}
pub fn representation_conversion_info(
    id: RepresentationConversionId,
) -> &'static RepresentationConversion {
    internals::representation_conversion_info(id)
}
/// The unique registered conversion whose source is `source`: the resident
/// form of an external representation. `None` for dense and packed
/// representations. The table build asserts at most one conversion per source.
pub fn resident_conversion(source: RepresentationId) -> Option<&'static RepresentationConversion> {
    internals::resident_conversion(source)
}
/// Canonical storage bytes of a contiguous tensor with `extents` in
/// representation `id`: dense = elements × dtype bytes; packed and external =
/// rows × ceil(last / group) × packet bytes, where rows is the product of the
/// leading extents. `None` when `extents` is empty for a packed or external
/// representation, or when the count overflows `u64`.
pub fn canonical_bytes(id: RepresentationId, extents: &[u64]) -> Option<u64> {
    let rows_and_last = || {
        let (last, leading) = extents.split_last()?;
        let rows = leading
            .iter()
            .try_fold(1u64, |rows, extent| rows.checked_mul(*extent))?;
        Some((rows, *last))
    };
    match &representation_info(id).kind {
        RepresentationKind::Dense(dtype) => extents
            .iter()
            .try_fold(u64::from(dtype.bytes()), |bytes, extent| bytes.checked_mul(*extent)),
        RepresentationKind::Packed(layout) => {
            let (rows, last) = rows_and_last()?;
            layout.bytes(rows, last)
        }
        RepresentationKind::External(layout) => {
            let (rows, last) = rows_and_last()?;
            rows.checked_mul(last.div_ceil(u64::from(layout.logical_group)))?
                .checked_mul(u64::from(layout.packet_size))
        }
    }
}
/// The dense representation of a dtype.
pub fn dense(dtype: DType) -> RepresentationId {
    internals::dense(dtype)
}

/// Storage dtype of one element of the plane view `t.<plane>` of a packed
/// representation (`PlaneInfo::storage_dtype`). The checker resolved both the
/// representation and the plane name, so either being unknown panics.
pub fn plane_element_dtype(representation: RepresentationId, plane: &str) -> DType {
    let info = representation_info(representation);
    let RepresentationKind::Packed(layout) = &info.kind else {
        panic!("plane view of the non-packed representation `{}`", info.name)
    };
    layout
        .planes
        .iter()
        .find(|candidate| candidate.name == plane)
        .unwrap_or_else(|| panic!("`{}` has no plane `{plane}`", info.name))
        .storage_dtype
}

/// Canonical typed decode recipe for a packed representation. Dense
/// representations need no recipe.
pub fn decode_recipe(id: RepresentationId, output: DType) -> Option<DecodeRecipe> {
    internals::packed(id).map(|representation| representation.decode_recipe_to(output))
}

/// Canonical round-to-nearest-even bfloat16 value used by the representation
/// registry and native ABI implementations.
pub fn bf16_round(value: f32) -> f32 {
    if value.is_nan() {
        return f32::from_bits((value.to_bits() | 0x0040_0000) & 0xffff_0000);
    }
    let bits = value.to_bits();
    let lsb = (bits >> 16) & 1;
    f32::from_bits(bits.wrapping_add(0x7fff + lsb) & 0xffff_0000)
}

/// Canonical round-to-nearest-even binary16 value.
pub fn f16_round(value: f32) -> f32 {
    if value.is_nan() || value.is_infinite() || value == 0.0 {
        return value;
    }
    let magnitude = value.abs();
    if magnitude >= 65_520.0 {
        return f32::INFINITY.copysign(value);
    }
    let bits = magnitude.to_bits();
    let exponent = ((bits >> 23) & 0xff) as i32 - 127;
    if exponent < -14 {
        let quantum = 2f32.powi(-24);
        return ((magnitude / quantum).round_ties_even() * quantum).copysign(value);
    }
    let lsb = (bits >> 13) & 1;
    let rounded = bits.wrapping_add((1 << 12) - 1 + lsb) & !((1 << 13) - 1);
    f32::from_bits(rounded).copysign(value)
}

/// Canonical binary16 payload.
pub fn f16_bits(value: f32) -> u16 {
    let value = f16_round(value);
    let bits = value.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exponent = ((bits >> 23) & 0xff) as i32;
    let mantissa = bits & 0x7f_ffff;
    if exponent == 0xff {
        return sign | 0x7c00 | if mantissa != 0 { 0x0200 } else { 0 };
    }
    let exponent = exponent - 127 + 15;
    if exponent >= 0x1f {
        return sign | 0x7c00;
    }
    if exponent <= 0 {
        if exponent < -10 {
            return sign;
        }
        return sign | (((mantissa | 0x80_0000) >> (1 - exponent + 13)) as u16);
    }
    sign | ((exponent as u16) << 10) | ((mantissa >> 13) as u16)
}

/// Exact widening of a binary16 payload to binary32.
pub fn f16_to_f32(value: u16) -> f32 {
    let sign = (u32::from(value & 0x8000)) << 16;
    let exponent = u32::from((value >> 10) & 0x1f);
    let mantissa = u32::from(value & 0x03ff);
    let bits = if exponent == 0 {
        if mantissa == 0 {
            sign
        } else {
            let mut normalized = mantissa;
            let mut adjustment = 0i32;
            while normalized & 0x0400 == 0 {
                normalized <<= 1;
                adjustment -= 1;
            }
            let mantissa = (normalized & 0x03ff) << 13;
            let exponent = (adjustment + 1 + 127 - 15) as u32;
            sign | (exponent << 23) | mantissa
        }
    } else if exponent == 0x1f {
        sign | 0x7f80_0000 | (mantissa << 13)
    } else {
        sign | ((exponent + 127 - 15) << 23) | (mantissa << 13)
    };
    f32::from_bits(bits)
}

pub(crate) mod internals {
    //! The interned tables. Built once from the crate-private
    //! `intrinsics::capability_rows` and `repr::REPRS` tables; every id is an
    //! index into these vectors and is valid for the process lifetime.
    //! Capabilities are contiguous per backend and intrinsics contiguous per
    //! capability so backend/capability slices are direct sub-slices.

    use super::*;
    use crate::intrinsics::{capability_rows, RowOperand, RowResult};
    use crate::repr::{Coefficients, PlaneEncoding, Repr, REPRS};
    use std::sync::OnceLock;

    pub(crate) struct Tables {
        capabilities: Vec<CapabilityInfo>,
        /// `[start, end)` into `capabilities` per backend, in `BackendName::ALL` order.
        capability_ranges: Vec<(usize, usize)>,
        intrinsics: Vec<IntrinsicSignature>,
        denotations: Vec<IntrinsicDenotation>,
        /// `[start, end)` into `intrinsics` per capability.
        intrinsic_ranges: Vec<(usize, usize)>,
        /// Every row of one capability sharing one name, in declaration order.
        overloads: Vec<(CapabilityId, &'static str, Vec<IntrinsicId>)>,
        representations: Vec<RepresentationInfo>,
        conversions: Vec<RepresentationConversion>,
    }

    fn build() -> Tables {
        // Representations: every dense dtype first (ordinal = dtype ordinal),
        // then the packed representations in `REPRS` order.
        let mut representations = Vec::new();
        for dtype in DType::ALL {
            representations.push(RepresentationInfo {
                id: RepresentationId::new(representations.len() as u32),
                name: dtype.name(),
                kind: RepresentationKind::Dense(dtype),
                decoded: dtype,
                access: RepresentationAccess::ReadWrite,
            });
        }
        for repr in REPRS {
            representations.push(RepresentationInfo {
                id: RepresentationId::new(representations.len() as u32),
                name: repr.name,
                kind: RepresentationKind::Packed(packed_layout(repr)),
                decoded: DType::F32,
                access: RepresentationAccess::ReadOnly,
            });
        }
        let external_specs = [
            ("gguf_q4_k", 144, 256, "q4k"),
            ("gguf_q5_k", 176, 256, "q5k"),
            ("gguf_q6_k", 210, 256, "q6k"),
            ("gguf_q8_0", 34, 32, "q8g32s"),
            ("gguf_iq4_xs", 136, 256, "iq4g32"),
        ];
        for (name, packet_size, logical_group, _) in external_specs {
            representations.push(RepresentationInfo {
                id: RepresentationId::new(representations.len() as u32),
                name,
                kind: RepresentationKind::External(ExternalPacketLayout {
                    packet_size,
                    packet_alignment: 1,
                    logical_group,
                    packing_axis: PackingAxis::Last,
                }),
                decoded: DType::F32,
                access: RepresentationAccess::ConversionSource,
            });
        }
        let mut conversions = Vec::new();
        for (source_name, _, _, destination_name) in external_specs {
            let source = representations
                .iter()
                .find(|representation| representation.name == source_name)
                .expect("registered external representation is present")
                .id;
            let destination = representations
                .iter()
                .find(|representation| representation.name == destination_name)
                .expect("registered resident representation is present")
                .id;
            let RepresentationKind::Packed(layout) = &representations[destination.index()].kind
            else {
                panic!("external conversion destination is not packed")
            };
            let recipe = external_repack_recipe(source_name, layout);
            validate_repack_recipe(
                representations[source.index()].kind.clone(),
                layout,
                &recipe,
            );
            assert!(
                conversions
                    .iter()
                    .all(|conversion: &RepresentationConversion| conversion.source != source),
                "external representation `{source_name}` has more than one resident conversion"
            );
            conversions.push(RepresentationConversion {
                id: RepresentationConversionId::new(conversions.len() as u32),
                source,
                destination,
                recipe,
            });
        }
        let dense_id = |dtype: DType| RepresentationId::new(u32::from(dtype.ordinal()));

        let rows = capability_rows();
        let mut capabilities: Vec<CapabilityInfo> = Vec::new();
        let mut capability_ranges = Vec::new();
        let mut intrinsics = Vec::new();
        let mut denotations = Vec::new();
        let mut intrinsic_ranges = Vec::new();
        let mut overloads: Vec<(CapabilityId, &'static str, Vec<IntrinsicId>)> = Vec::new();
        for backend in BackendName::ALL {
            let start = capabilities.len();
            for row in rows.iter().filter(|row| row.backend == backend) {
                let capability = match capabilities
                    .iter()
                    .position(|c| c.backend == backend && c.name == row.capability)
                {
                    Some(index) => capabilities[index].id,
                    None => {
                        let id = CapabilityId::new(capabilities.len() as u32);
                        capabilities.push(CapabilityInfo {
                            id,
                            backend,
                            name: row.capability,
                        });
                        intrinsic_ranges.push((intrinsics.len(), intrinsics.len()));
                        id
                    }
                };
                let id = IntrinsicId::new(intrinsics.len() as u32);
                let result = match row.result {
                    RowResult::Scalar(dtype) => IntrinsicResultType::Scalar(dtype),
                    RowResult::Owned(dtype, axes) => IntrinsicResultType::Owned {
                        representation: dense_id(dtype),
                        axes,
                    },
                };
                if let IntrinsicResultType::Owned { axes, .. } = &result {
                    for projection in *axes {
                        let (_, operand) = row.arguments.get(projection.argument as usize)
                            .expect("result axis selects an actual intrinsic argument");
                        let rank = match operand {
                            RowOperand::Readable(_, rank) | RowOperand::ReadableRepresentation(_, rank) => *rank,
                            RowOperand::Scalar(_) => panic!("result axis cannot select a scalar argument"),
                        };
                        assert!(projection.axis < rank, "result axis lies within its argument rank");
                    }
                }
                if let IntrinsicExecution::WholeTensor {
                    result: result_index,
                } = row.execution
                {
                    assert_eq!(result_index, 0, "intrinsic has exactly one result");
                    assert!(matches!(result, IntrinsicResultType::Owned { .. }));
                }
                intrinsics.push(IntrinsicSignature {
                    id,
                    capability,
                    name: row.name,
                    arguments: row
                        .arguments
                        .iter()
                        .map(|(name, operand)| IntrinsicArgument {
                            name,
                            category: match *operand {
                                RowOperand::Scalar(dtype) => OperandCategory::Scalar(dtype),
                                RowOperand::Readable(dtype, rank) => OperandCategory::Readable {
                                    representation: dense_id(dtype),
                                    rank,
                                },
                                RowOperand::ReadableRepresentation(name, rank) => {
                                    let representation = representations
                                        .iter()
                                        .find(|representation| representation.name == name)
                                        .unwrap_or_else(|| panic!("intrinsic row names unknown representation `{name}`"))
                                        .id;
                                    OperandCategory::Readable {
                                        representation,
                                        rank,
                                    }
                                }
                            },
                        })
                        .collect(),
                    result,
                    execution: row.execution,
                    effects: IntrinsicEffects {
                        writes: Vec::new(),
                        participation: row.participation,
                        result_uniformity: row.result_uniformity,
                    },
                    numerical: row.numerics.clone(),
                });
                denotations.push(row.denotation);
                intrinsic_ranges[capability.index()].1 = intrinsics.len();
                match overloads
                    .iter_mut()
                    .find(|(owner, name, _)| *owner == capability && *name == row.name)
                {
                    Some((_, _, rows)) => {
                        let categories = |row: IntrinsicId| {
                            intrinsics[row.index()]
                                .arguments
                                .iter()
                                .map(|argument| &argument.category)
                                .collect::<Vec<_>>()
                        };
                        assert!(
                            rows.iter().all(|other| categories(*other) != categories(id)),
                            "two rows of one intrinsic overload declare the same operands"
                        );
                        rows.push(id);
                    }
                    None => overloads.push((capability, row.name, vec![id])),
                }
            }
            capability_ranges.push((start, capabilities.len()));
        }
        Tables {
            capabilities,
            capability_ranges,
            intrinsics,
            denotations,
            intrinsic_ranges,
            overloads,
            representations,
            conversions,
        }
    }

    /// The physical planes of one packed representation, per storage group.
    fn packed_layout(repr: &Repr) -> PackedPacketLayout {
        let group = u64::from(repr.storage_group());
        let mut offset = 0u64;
        let mut packet_alignment = 1u32;
        let mut planes = Vec::new();
        for plane in repr.planes() {
            let plane_group = u64::from(plane.group);
            assert!(
                group.is_multiple_of(plane_group),
                "packed plane group must divide the representation packet group"
            );
            let bits = u64::from(plane.entry_bits())
                .checked_mul(u64::from(plane.fields))
                .and_then(|value| value.checked_mul(group / plane_group))
                .unwrap_or_else(|| panic!("packed representation plane size overflow"));
            let alignment = match plane.encoding {
                PlaneEncoding::Dense(dtype) => dtype.bytes(),
                PlaneEncoding::Packed { .. } => 4,
                PlaneEncoding::FloatCode { .. } => 1,
            };
            let alignment64 = u64::from(alignment);
            offset = offset
                .checked_add(alignment64 - 1)
                .and_then(|value| value.checked_div(alignment64))
                .and_then(|value| value.checked_mul(alignment64))
                .unwrap_or_else(|| panic!("packed representation packet layout overflow"));
            let bytes_per_group = bits.div_ceil(8);
            planes.push(PlaneInfo {
                name: plane.name,
                offset: u32::try_from(offset).expect("packed plane offset exceeds u32::MAX"),
                bytes_per_group: u32::try_from(bytes_per_group)
                    .expect("packed plane exceeds u32::MAX bytes per group"),
                alignment,
                encoding: plane.encoding.clone(),
                group: plane.group,
                fields: plane.fields,
                entry_bits: plane.entry_bits(),
                storage_dtype: plane.dtype(),
            });
            offset = offset
                .checked_add(bytes_per_group)
                .unwrap_or_else(|| panic!("packed representation packet layout overflow"));
            packet_alignment = packet_alignment.max(alignment);
        }
        let alignment = u64::from(packet_alignment);
        let packet_size = offset
            .checked_add(alignment - 1)
            .and_then(|value| value.checked_div(alignment))
            .and_then(|value| value.checked_mul(alignment))
            .unwrap_or_else(|| panic!("packed representation packet layout overflow"));
        PackedPacketLayout {
            planes,
            group: repr.storage_group(),
            packing_axis: PackingAxis::Last,
            packet_size: u32::try_from(packet_size).expect("packed packet exceeds u32::MAX bytes"),
            packet_alignment,
        }
    }

    fn direct_bits(byte: u32, bytes: u32) -> Vec<u32> {
        (0..bytes * 8).map(|bit| byte * 8 + bit).collect()
    }

    fn hierarchical_coefficients() -> Vec<u32> {
        (0..96)
            .map(|destination_bit| {
                let field = destination_bit / 6;
                let group = field / 2;
                let local_bit = destination_bit % 6;
                let low_byte = 4 + (field % 2) * 4 + group % 4;
                if group < 4 || local_bit >= 4 {
                    low_byte * 8 + if group < 4 { local_bit } else { local_bit + 2 }
                } else {
                    let high_byte = 12 + group % 4;
                    high_byte * 8 + (field % 2) * 4 + local_bit
                }
            })
            .collect()
    }

    fn external_repack_recipe(
        source_name: &str,
        destination: &PackedPacketLayout,
    ) -> PacketRepackRecipe {
        let plane = |name: &str, recipe: PlaneRepackRecipe| {
            let index = destination
                .planes
                .iter()
                .position(|plane| plane.name == name)
                .unwrap_or_else(|| panic!("resident representation has no `{name}` plane"));
            (index, recipe)
        };
        let mut recipes = Vec::new();
        let mut push = |entry: (usize, PlaneRepackRecipe)| recipes.push(entry);
        match source_name {
            "gguf_q4_k" => {
                push(plane(
                    "words",
                    PlaneRepackRecipe::BitRoutes(
                        (0..1024)
                            .map(|bit| {
                                let position = bit / 4;
                                let source_byte = 16 + (position / 64) * 32 + position % 32;
                                source_byte * 8 + (position % 64 / 32) * 4 + bit % 4
                            })
                            .collect(),
                    ),
                ));
                push(plane(
                    "coefficients",
                    PlaneRepackRecipe::BitRoutes(hierarchical_coefficients()),
                ));
                push(plane(
                    "scale_factor",
                    PlaneRepackRecipe::BitRoutes(direct_bits(0, 2)),
                ));
                push(plane(
                    "bias_factor",
                    PlaneRepackRecipe::BitRoutes(direct_bits(2, 2)),
                ));
            }
            "gguf_q5_k" => {
                push(plane(
                    "words",
                    PlaneRepackRecipe::BitRoutes(
                        (0..1280)
                            .map(|bit| {
                                let position = bit / 5;
                                let code_bit = bit % 5;
                                if code_bit < 4 {
                                    let source_byte = 48 + (position / 64) * 32 + position % 32;
                                    source_byte * 8 + (position % 64 / 32) * 4 + code_bit
                                } else {
                                    let source_byte = 16 + position % 32;
                                    source_byte * 8 + position / 32
                                }
                            })
                            .collect(),
                    ),
                ));
                push(plane(
                    "coefficients",
                    PlaneRepackRecipe::BitRoutes(hierarchical_coefficients()),
                ));
                push(plane(
                    "scale_factor",
                    PlaneRepackRecipe::BitRoutes(direct_bits(0, 2)),
                ));
                push(plane(
                    "bias_factor",
                    PlaneRepackRecipe::BitRoutes(direct_bits(2, 2)),
                ));
            }
            "gguf_q6_k" => {
                push(plane(
                    "words",
                    PlaneRepackRecipe::BitRoutes(
                        (0..1536)
                            .map(|bit| {
                                let position = bit / 6;
                                let code_bit = bit % 6;
                                if code_bit < 4 {
                                    let source_byte = (position / 128) * 64 + position % 64;
                                    source_byte * 8 + (position % 128 / 64) * 4 + code_bit
                                } else {
                                    let source_byte = 128 + (position / 128) * 32 + position % 32;
                                    source_byte * 8 + (position % 128 / 32) * 2 + code_bit - 4
                                }
                            })
                            .collect(),
                    ),
                ));
                push(plane(
                    "coefficients",
                    PlaneRepackRecipe::BitRoutes(direct_bits(192, 16)),
                ));
                push(plane(
                    "scale_factor",
                    PlaneRepackRecipe::BitRoutes(direct_bits(208, 2)),
                ));
            }
            "gguf_q8_0" => {
                push(plane(
                    "words",
                    PlaneRepackRecipe::BitRoutes(direct_bits(2, 32)),
                ));
                push(plane(
                    "scale",
                    PlaneRepackRecipe::BitRoutes(direct_bits(0, 2)),
                ));
            }
            "gguf_iq4_xs" => {
                push(plane(
                    "words",
                    PlaneRepackRecipe::BitRoutes(
                        (0..1024)
                            .map(|bit| {
                                let position = bit / 4;
                                let source_byte = 8 + (position / 32) * 16 + position % 16;
                                source_byte * 8 + (position % 32 / 16) * 4 + bit % 4
                            })
                            .collect(),
                    ),
                ));
                let base =
                    RepackExpr::F16ToF32(Box::new(RepackExpr::SourceBits { bit: 0, width: 16 }));
                push(plane(
                    "scale",
                    PlaneRepackRecipe::DenseValues(
                        (0..8)
                            .map(|group| {
                                let combined = RepackExpr::OffsetI32 {
                                    value: Box::new(RepackExpr::BitOr(
                                        Box::new(RepackExpr::SourceBits {
                                            bit: (4 + group / 2) * 8 + (group % 2) * 4,
                                            width: 4,
                                        }),
                                        Box::new(RepackExpr::ShiftLeft {
                                            value: Box::new(RepackExpr::SourceBits {
                                                bit: 16 + 2 * group,
                                                width: 2,
                                            }),
                                            bits: 4,
                                        }),
                                    )),
                                    offset: -32,
                                };
                                RepackExpr::MultiplyF32(
                                    Box::new(base.clone()),
                                    Box::new(RepackExpr::I32ToF32(Box::new(combined))),
                                )
                            })
                            .collect(),
                    ),
                ));
            }
            _ => panic!("unknown external representation `{source_name}`"),
        }
        recipes.sort_by_key(|(index, _)| *index);
        assert_eq!(recipes.len(), destination.planes.len());
        PacketRepackRecipe {
            planes: recipes.into_iter().map(|(_, recipe)| recipe).collect(),
        }
    }

    fn validate_repack_recipe(
        source: RepresentationKind,
        destination: &PackedPacketLayout,
        recipe: &PacketRepackRecipe,
    ) {
        let RepresentationKind::External(source) = source else {
            panic!("representation conversion source is not external storage")
        };
        assert!(source.packet_size > 0, "external packet size is zero");
        assert!(
            source.packet_alignment > 0
                && source.packet_alignment.is_power_of_two()
                && source.packet_size % source.packet_alignment == 0,
            "external packet alignment is invalid"
        );
        assert_eq!(
            source.logical_group, destination.group,
            "representation conversion is not one logical packet to one resident packet"
        );
        assert_eq!(
            source.packing_axis, destination.packing_axis,
            "representation conversion changes its packing axis"
        );
        assert_eq!(
            recipe.planes.len(),
            destination.planes.len(),
            "representation conversion does not initialize every destination plane"
        );
        let source_bits = source
            .packet_size
            .checked_mul(8)
            .expect("external packet bit count exceeds u32");
        fn validate_expr(expression: &RepackExpr, source_bits: u32) {
            match expression {
                RepackExpr::SourceBits { bit, width } => {
                    let width = u32::from(*width);
                    assert!(
                        (1..=32).contains(&width),
                        "representation conversion source word is not representable as u32"
                    );
                    assert!(
                        bit.checked_add(width).is_some_and(|end| end <= source_bits),
                        "representation conversion reads beyond its source packet"
                    );
                }
                RepackExpr::ShiftLeft { value, .. }
                | RepackExpr::OffsetI32 { value, .. }
                | RepackExpr::F16ToF32(value)
                | RepackExpr::I32ToF32(value) => validate_expr(value, source_bits),
                RepackExpr::BitOr(left, right) | RepackExpr::MultiplyF32(left, right) => {
                    validate_expr(left, source_bits);
                    validate_expr(right, source_bits);
                }
            }
        }
        for (plane, recipe) in destination.planes.iter().zip(&recipe.planes) {
            let plane_bits = plane
                .bytes_per_group
                .checked_mul(8)
                .expect("destination plane bit count exceeds u32");
            match recipe {
                PlaneRepackRecipe::BitRoutes(routes) => {
                    assert_eq!(
                        u32::try_from(routes.len()).expect("plane bit count exceeds u32"),
                        plane_bits,
                        "bit-route conversion does not initialize its entire plane"
                    );
                    assert!(
                        routes.iter().all(|source| *source < source_bits),
                        "bit-route conversion reads beyond its source packet"
                    );
                }
                PlaneRepackRecipe::DenseValues(values) => {
                    let PlaneEncoding::Dense(dtype) = &plane.encoding else {
                        panic!("dense conversion recipe targets a non-dense plane")
                    };
                    let elements = plane.bytes_per_group / dtype.bytes();
                    assert_eq!(
                        u32::try_from(values.len()).expect("plane element count exceeds u32"),
                        elements,
                        "dense conversion does not initialize its entire plane"
                    );
                    for expression in values {
                        validate_expr(expression, source_bits);
                    }
                }
            }
        }
    }

    pub(crate) fn tables() -> &'static Tables {
        static TABLES: OnceLock<Tables> = OnceLock::new();
        TABLES.get_or_init(build)
    }

    pub(super) fn capability(backend: BackendName, name: &str) -> Option<CapabilityId> {
        capabilities(backend)
            .iter()
            .find(|c| c.name == name)
            .map(|c| c.id)
    }
    pub(super) fn capability_info(id: CapabilityId) -> &'static CapabilityInfo {
        tables().capabilities.get(id.index()).unwrap_or_else(|| {
            panic!("StaticRegistry produced a CapabilityId outside its capability table (§13.3.1)")
        })
    }
    pub(super) fn capabilities(backend: BackendName) -> &'static [CapabilityInfo] {
        let t = tables();
        let (start, end) = t.capability_ranges[backend as usize];
        &t.capabilities[start..end]
    }
    pub(super) fn intrinsic_overloads(capability: CapabilityId, name: &str) -> &'static [IntrinsicId] {
        tables()
            .overloads
            .iter()
            .find(|(owner, member, _)| *owner == capability && *member == name)
            .map_or(&[], |(_, _, rows)| rows.as_slice())
    }
    pub(super) fn intrinsic_signature(id: IntrinsicId) -> &'static IntrinsicSignature {
        tables().intrinsics.get(id.index()).unwrap_or_else(|| {
            panic!("StaticRegistry produced an IntrinsicId outside its intrinsic table (§13.3.1)")
        })
    }
    pub(super) fn intrinsics(capability: CapabilityId) -> &'static [IntrinsicSignature] {
        let t = tables();
        let (start, end) = *t
            .intrinsic_ranges
            .get(capability.index())
            .unwrap_or_else(|| {
                panic!(
                    "StaticRegistry produced a CapabilityId outside its intrinsic ranges (§13.3.1)"
                )
            });
        &t.intrinsics[start..end]
    }
    pub(super) fn representation(name: &str) -> Option<RepresentationId> {
        representations()
            .iter()
            .find(|r| r.name == name)
            .map(|r| r.id)
    }
    pub(super) fn representation_info(id: RepresentationId) -> &'static RepresentationInfo {
        tables().representations.get(id.index()).unwrap_or_else(|| {
            panic!("StaticRegistry produced a RepresentationId outside its representation table (§13.3.1)")
        })
    }
    pub(super) fn representations() -> &'static [RepresentationInfo] {
        &tables().representations
    }
    pub(super) fn representation_conversion(
        source: RepresentationId,
        destination: RepresentationId,
    ) -> Option<&'static RepresentationConversion> {
        tables()
            .conversions
            .iter()
            .find(|conversion| conversion.source == source && conversion.destination == destination)
    }
    pub(super) fn resident_conversion(
        source: RepresentationId,
    ) -> Option<&'static RepresentationConversion> {
        tables()
            .conversions
            .iter()
            .find(|conversion| conversion.source == source)
    }
    pub(super) fn representation_conversion_info(
        id: RepresentationConversionId,
    ) -> &'static RepresentationConversion {
        tables().conversions.get(id.index()).unwrap_or_else(|| {
            panic!("StaticRegistry produced a RepresentationConversionId outside its conversion table (§13.3.1)")
        })
    }
    pub(super) fn dense(dtype: DType) -> RepresentationId {
        RepresentationId::new(u32::from(dtype.ordinal()))
    }

    pub(super) fn intrinsic_denotation(id: IntrinsicId) -> IntrinsicDenotation {
        *tables().denotations.get(id.index()).unwrap_or_else(|| {
            panic!("StaticRegistry produced an IntrinsicId outside its denotation table (§13.3.1)")
        })
    }

    // ----- crate-private views -------------------------------------------

    /// Every intrinsic signature, in id order.
    pub(crate) fn all_intrinsics() -> &'static [IntrinsicSignature] {
        &tables().intrinsics
    }

    /// The packed representation behind an id, when it is not dense.
    pub(crate) fn packed(id: RepresentationId) -> Option<&'static Repr> {
        REPRS.get(id.index().checked_sub(DType::ALL.len())?)
    }

    /// The dense dtype behind an id, when it is not packed.
    pub(crate) fn dense_dtype(id: RepresentationId) -> Option<DType> {
        match representation_info(id).kind {
            RepresentationKind::Dense(dtype) => Some(dtype),
            RepresentationKind::Packed(_) | RepresentationKind::External(_) => None,
        }
    }

    /// Whether a representation carries a bias plane (used by the reference
    /// decoder).
    pub(crate) fn has_bias(repr: &Repr) -> bool {
        match repr.coefficients {
            Coefficients::Direct { bias, .. } | Coefficients::Hierarchical { bias, .. } => bias,
            Coefficients::BlockFloat { .. } => false,
        }
    }

    /// Stable wire name of an intrinsic: `backend.capability.name#ordinal`,
    /// where `ordinal` disambiguates overloads of one name within a
    /// capability.
    pub(crate) fn wire_name(id: IntrinsicId) -> String {
        let signature = intrinsic_signature(id);
        let capability = capability_info(signature.capability);
        let ordinal = intrinsics(signature.capability)
            .iter()
            .filter(|s| s.name == signature.name && s.id < id)
            .count();
        format!(
            "{}.{}.{}#{ordinal}",
            capability.backend.as_str(),
            capability.name,
            signature.name
        )
    }

    /// The intrinsic with a wire name, at decode time.
    pub(crate) fn by_wire_name(name: &str) -> Option<IntrinsicId> {
        let (path, ordinal) = name.rsplit_once('#')?;
        let ordinal: usize = ordinal.parse().ok()?;
        let mut parts = path.splitn(3, '.');
        let backend = BackendName::parse(parts.next()?)?;
        let capability = capability(backend, parts.next()?)?;
        let member = parts.next()?;
        intrinsics(capability)
            .iter()
            .filter(|s| s.name == member)
            .nth(ordinal)
            .map(|s| s.id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tables_intern_every_row_once() {
        let rows = crate::intrinsics::capability_rows();
        assert_eq!(internals::all_intrinsics().len(), rows.len());
        for backend in BackendName::ALL {
            for info in capabilities(backend) {
                assert_eq!(info.backend, backend);
                assert_eq!(capability(backend, info.name), Some(info.id));
                for signature in intrinsics(info.id) {
                    assert_eq!(signature.capability, info.id);
                    assert_eq!(
                        internals::by_wire_name(&internals::wire_name(signature.id)),
                        Some(signature.id)
                    );
                }
            }
        }
        assert_eq!(
            representation("q4g64")
                .map(representation_info)
                .map(|r| r.decoded),
            Some(DType::F32)
        );
        assert_eq!(representation_info(dense(DType::BF16)).name, "bf16");
    }

    #[test]
    fn overloads_resolve_each_row_by_its_operands() {
        for backend in BackendName::ALL {
            for info in capabilities(backend) {
                for signature in intrinsics(info.id) {
                    let overload = intrinsic_overloads(info.id, signature.name);
                    assert!(overload.contains(&signature.id));
                    assert!(overload.windows(2).all(|pair| pair[0] < pair[1]));
                    let operands = signature
                        .arguments
                        .iter()
                        .map(|argument| match argument.category {
                            OperandCategory::Scalar(dtype) => OperandElement::Scalar(dtype),
                            OperandCategory::Readable {
                                representation,
                                rank,
                            } => OperandElement::Tensor {
                                representation,
                                rank,
                            },
                            _ => panic!("row declares a non-readable operand"),
                        })
                        .collect::<Vec<_>>();
                    assert_eq!(resolve_intrinsic(overload, &operands), Some(signature.id));
                }
                assert!(intrinsic_overloads(info.id, "no_such_intrinsic").is_empty());
            }
        }
        let matrix = capability(BackendName::Metal, "matrix").unwrap();
        let matmul = intrinsic_overloads(matrix, "matmul");
        assert!(matmul.len() > 1);
        let f16 = OperandElement::Tensor {
            representation: dense(DType::F16),
            rank: 2,
        };
        let resolved = resolve_intrinsic(matmul, &[f16, f16]).unwrap();
        assert_eq!(
            intrinsic_denotation(resolved),
            IntrinsicDenotation::MatrixProduct { accumulate: false }
        );
        assert_eq!(resolve_intrinsic(matmul, &[f16]), None);
        assert_eq!(
            resolve_intrinsic(matmul, &[OperandElement::Scalar(DType::F16), f16]),
            None
        );
    }

    #[test]
    fn denotations_and_participation_follow_the_rows() {
        let subgroup = capability(BackendName::Metal, "subgroup").unwrap();
        let sum = intrinsic_overloads(subgroup, "simd_sum")[0];
        assert_eq!(
            intrinsic_denotation(sum),
            IntrinsicDenotation::CohortFold {
                op: crate::intrinsics::ReduceOp::Sum
            }
        );
        let lane = intrinsic_overloads(subgroup, "lane_index")[0];
        assert_eq!(intrinsic_denotation(lane), IntrinsicDenotation::ParticipantIndex);
        let cuda_matrix = capability(BackendName::Cuda, "matrix").unwrap();
        for name in ["nvfp4_matmul", "nvfp4_matmul_add"] {
            for id in intrinsic_overloads(cuda_matrix, name) {
                assert_eq!(
                    intrinsic_signature(*id).effects.participation,
                    IntrinsicParticipation::FixedWorkgroup(128)
                );
            }
        }
    }

    #[test]
    fn plane_element_dtype_is_the_plane_storage_dtype() {
        for info in representations() {
            if let RepresentationKind::Packed(layout) = &info.kind {
                for plane in &layout.planes {
                    assert_eq!(plane_element_dtype(info.id, plane.name), plane.storage_dtype);
                }
            }
        }
    }

    #[test]
    fn direct_native_is_metal_only() {
        assert!(BackendName::Metal.supports_direct_native());
        assert!(!BackendName::Cpu.supports_direct_native());
        assert!(!BackendName::Cuda.supports_direct_native());
    }

    #[test]
    fn narrow_float_encodings_cover_subnormals_and_finite_roundtrips() {
        assert_eq!(f16_to_f32(1), 2.0f32.powi(-24));
        assert_eq!(f16_to_f32(0x03ff), 1023.0 * 2.0f32.powi(-24));
        for bits in 0..=u16::MAX {
            let value = f16_to_f32(bits);
            if !value.is_nan() {
                assert_eq!(f16_bits(value), bits, "half encoding {bits:04x}");
            }
        }
    }

    #[test]
    fn narrow_nan_never_encodes_as_infinity() {
        for bits in [0x7f80_0001, 0xff80_0001, 0x7fc0_0000, 0x7fff_ffff] {
            let value = f32::from_bits(bits);
            let rounded = bf16_round(value);
            assert!(rounded.is_nan());
            assert_eq!(rounded.to_bits() & 0xffff, 0);
            assert!(f16_to_f32(f16_bits(value)).is_nan());
        }
    }

    fn packed_layout(name: &str) -> &'static PackedPacketLayout {
        let RepresentationKind::Packed(layout) =
            &representation_info(representation(name).unwrap()).kind
        else {
            panic!("`{name}` is packed")
        };
        layout
    }

    #[test]
    fn resident_conversion_is_the_unique_conversion_of_an_external_source() {
        let q4k = representation_conversion_info(
            resident_conversion(representation("gguf_q4_k").unwrap()).unwrap().id,
        );
        assert_eq!(q4k.destination, representation("q4k").unwrap());
        for info in representations() {
            let conversion = resident_conversion(info.id);
            match info.kind {
                RepresentationKind::External(_) => {
                    let conversion = conversion.unwrap();
                    assert_eq!(conversion.source, info.id);
                    assert_eq!(
                        representation_conversion(info.id, conversion.destination),
                        Some(conversion)
                    );
                }
                RepresentationKind::Dense(_) | RepresentationKind::Packed(_) => {
                    assert_eq!(conversion, None)
                }
            }
        }
    }

    #[test]
    fn canonical_bytes_follow_the_representation_layout() {
        let q4g64 = representation("q4g64").unwrap();
        let q4_k = representation("gguf_q4_k").unwrap();
        assert_eq!(canonical_bytes(dense(DType::F32), &[3, 5]), Some(60));
        assert_eq!(canonical_bytes(dense(DType::BF16), &[]), Some(2));
        assert_eq!(canonical_bytes(dense(DType::F32), &[0, 5]), Some(0));
        assert_eq!(canonical_bytes(q4g64, &[3, 128]), Some(3 * 2 * 36));
        assert_eq!(canonical_bytes(q4g64, &[2, 3, 65]), Some(6 * 2 * 36));
        assert_eq!(canonical_bytes(q4_k, &[512]), Some(288));
        assert_eq!(canonical_bytes(q4_k, &[2, 257]), Some(2 * 2 * 144));
        assert_eq!(canonical_bytes(q4g64, &[]), None);
        assert_eq!(canonical_bytes(q4_k, &[]), None);
        assert_eq!(canonical_bytes(dense(DType::F32), &[u64::MAX, 2]), None);
        assert_eq!(canonical_bytes(q4g64, &[u64::MAX, 64]), None);
    }

    #[test]
    fn packets_from_planes_places_each_plane_at_its_packet_offset() {
        for info in representations() {
            let RepresentationKind::Packed(layout) = &info.kind else {
                continue;
            };
            let (rows, extent) = (2, u64::from(layout.group) * 2 - 1);
            let packets = (rows * layout.packet_extent(extent)) as usize;
            let supplied: Vec<(&str, Vec<u8>)> = layout
                .planes
                .iter()
                .enumerate()
                .map(|(index, plane)| {
                    let bytes = (0..packets * plane.bytes_per_group as usize)
                        .map(|byte| (byte * 7 + index * 31 + 1) as u8)
                        .collect();
                    (plane.name, bytes)
                })
                .collect();
            let reversed: Vec<(&str, &[u8])> = supplied
                .iter()
                .rev()
                .map(|(name, bytes)| (*name, bytes.as_slice()))
                .collect();
            let canonical = layout.packets_from_planes(rows, extent, &reversed).unwrap();
            assert_eq!(Some(canonical.len() as u64), layout.bytes(rows, extent));
            let mut occupied = vec![false; canonical.len()];
            for (index, (plane, (_, bytes))) in layout.planes.iter().zip(&supplied).enumerate() {
                let width = plane.bytes_per_group as usize;
                for packet in 0..packets {
                    let start = layout.plane_offset(packet as u64, index).unwrap() as usize;
                    assert_eq!(
                        &canonical[start..start + width],
                        &bytes[packet * width..(packet + 1) * width],
                        "`{}` plane `{}` packet {packet}",
                        info.name,
                        plane.name
                    );
                    occupied[start..start + width].fill(true);
                }
            }
            assert!(canonical
                .iter()
                .zip(&occupied)
                .all(|(byte, occupied)| *occupied || *byte == 0));
        }
    }

    #[test]
    fn packets_from_planes_refuses_a_wrong_plane_set_or_length() {
        let layout = packed_layout("q4g64");
        let words = [0u8; 4 * 32];
        let scale = [0u8; 4 * 2];
        let bias = [0u8; 4 * 2];
        let assemble = |planes: &[(&str, &[u8])]| layout.packets_from_planes(2, 128, planes);
        assert_eq!(
            assemble(&[("words", &words), ("scale", &scale)]),
            Err(PlaneAssemblyError::PlaneSet)
        );
        assert_eq!(
            assemble(&[("words", &words), ("scale", &scale), ("scale", &bias)]),
            Err(PlaneAssemblyError::PlaneSet)
        );
        assert_eq!(
            assemble(&[("words", &words), ("scale", &scale), ("bias", &bias), ("extra", &bias)]),
            Err(PlaneAssemblyError::PlaneSet)
        );
        assert_eq!(
            assemble(&[("words", &words), ("scale", &scale[..6]), ("bias", &bias)]),
            Err(PlaneAssemblyError::PlaneByteLength {
                plane: "scale",
                expected: 8,
                actual: 6
            })
        );
        assert_eq!(
            assemble(&[("bias", &bias), ("words", &words), ("scale", &scale)])
                .map(|canonical| canonical.len()),
            Ok(4 * 36)
        );
    }
}
