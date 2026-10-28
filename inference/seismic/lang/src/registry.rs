//! Typed registry identities (spec §3.3, §4.2, §7.1).
//!
//! Capabilities, intrinsic signatures, and representations are interned
//! typed ids, never strings. The registry is static and sealed: every id is
//! valid for the lifetime of the process, and lookups by name exist only at
//! the source-checking and target-profiling boundaries.
//!
//! W1 owns the interning over the existing `intrinsics` and `repr` tables.

pub use crate::external_tables::RepackTable;
use crate::ids::{CapabilityId, IntrinsicId, RepresentationConversionId, RepresentationId};
use crate::types::DType;
use magnitude_ggml::Encoding;

pub use crate::repr::{
    CodeInterpretation, CodeKind, CoefficientAxes, CoefficientKind, DecodeAxes, DecodeComponent,
    DecodeRecipe, DecodeStep, DecodeTemp, FloatCodeFormat, GridCodes, GridSigns, LocalCodes,
    LookupTable, MinimumSign, OperandRecipe, PlaneEncoding, PlaneField, PlaneRead, PlaneSchema,
    QuantizedComponent, ScaleEncoding, ScaledCodes, GRID_VALUES,
};

/// Backends are a closed set.
#[derive(
    Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, serde::Serialize, serde::Deserialize,
)]
pub enum BackendName {
    Cpu,
    Metal,
    Cuda,
    /// Native-only: authored `native … for vulkan` implementations run on it;
    /// it has no compiler target, capabilities or intrinsics.
    Vulkan,
}

impl BackendName {
    pub const ALL: [BackendName; 4] = [
        BackendName::Cpu,
        BackendName::Metal,
        BackendName::Cuda,
        BackendName::Vulkan,
    ];

    pub fn parse(name: &str) -> Option<BackendName> {
        match name {
            "cpu" => Some(BackendName::Cpu),
            "metal" => Some(BackendName::Metal),
            "cuda" => Some(BackendName::Cuda),
            "vulkan" => Some(BackendName::Vulkan),
            _ => None,
        }
    }

    pub const fn as_str(self) -> &'static str {
        match self {
            BackendName::Cpu => "cpu",
            BackendName::Metal => "metal",
            BackendName::Cuda => "cuda",
            BackendName::Vulkan => "vulkan",
        }
    }

    /// The row layout this backend's native kernels read resident packed
    /// weights in: Metal `rows32`, CUDA `mma16`, Vulkan `rows16`, the CPU
    /// `rows8`.
    pub const fn resident_layout(self) -> Layout {
        match self {
            BackendName::Cpu => Layout::Rows8,
            BackendName::Metal => Layout::Rows32,
            BackendName::Cuda => Layout::Mma16,
            BackendName::Vulkan => Layout::Rows16,
        }
    }

    /// The dialect of the backend's native sources; `None` for the CPU,
    /// whose native implementations are Rust.
    pub const fn source_dialect(self) -> Option<SourceDialect> {
        match self {
            BackendName::Cpu => None,
            BackendName::Metal => Some(SourceDialect::Metal),
            BackendName::Cuda => Some(SourceDialect::Cuda),
            BackendName::Vulkan => Some(SourceDialect::Vulkan),
        }
    }

    /// Whether the backend is a compiler target for planned code (portable
    /// bodies and `lower … for` bodies). A native-only backend runs only
    /// authored `native` implementations.
    pub const fn compiles_planned_code(self) -> bool {
        match self {
            BackendName::Cpu | BackendName::Metal | BackendName::Cuda => true,
            BackendName::Vulkan => false,
        }
    }
}

/// The source dialect of a backend's native implementations: the sources a
/// build validates against their ABI and the runtime renders a prefix for
/// (MSL, CUDA C++, GLSL). A CPU native implementation is Rust compiled into
/// the embedding crate, so it has none ([`BackendName::source_dialect`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum SourceDialect {
    Metal,
    Cuda,
    Vulkan,
}

/// Revision of the whole registry. Any semantic change to a primitive,
/// capability, intrinsic, or representation changes this string, and with it
/// every cache identity.
pub const REGISTRY_REVISION: &str = "seismic-registry-v17";

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

/// The byte arrangement of a packed representation (spec S1). A packed
/// tensor's storage type is the pair (representation, layout): the
/// representation defines logical values (codes, coefficients, decode
/// formula); the layout defines where their bytes live. Dense and external
/// storage is always `Packet` (one storage unit per element or source packet).
///
/// - `Packet`: every packet interleaves its planes (`PackedPacketLayout`).
///   The portable oracle layout.
/// - `Rows16`: row-major; each row holds its planes one after another, every
///   plane and every row 16 B-aligned, codes split into a low-4-bit plane and
///   a high-bit plane (`PackedRowLayout`). The Metal resident layout.
/// - `Rows8`: eight-row tiles, with each plane interleaved by storage group
///   across the tile. The CPU resident layout.
/// - `Mma16`: `Rows16` whose code-plane bytes are permuted, per 16-row tile,
///   into `mma.sync.m16n8k16` A-fragment order (`PackedRowLayout::code_bit`).
///   The CUDA resident layout.
/// - `Rows32`: 32-row tiles (one row per lane of a 32-lane cooperative
///   multiply). Each code plane is interleaved across the tile per 32 columns,
///   so one aligned load is one row's codes of one packet and a tile's loads
///   are contiguous. Only code planes are interleaved, because the multiply
///   reads them a tile at a time; every other plane holds the tile's rows
///   one after another, each row's bytes contiguous as in `Rows16`, because
///   every consumer reads coefficients a row at a time.
///   The Metal resident layout (`inference/docs/kernels/lane-order.md`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Layout {
    Packet,
    Rows16,
    Rows8,
    Mma16,
    Rows32,
}

impl Layout {
    pub const ALL: [Layout; 5] = [
        Layout::Packet,
        Layout::Rows16,
        Layout::Rows8,
        Layout::Mma16,
        Layout::Rows32,
    ];

    pub const fn as_str(self) -> &'static str {
        match self {
            Layout::Packet => "packet",
            Layout::Rows16 => "rows16",
            Layout::Rows8 => "rows8",
            Layout::Mma16 => "mma16",
            Layout::Rows32 => "rows32",
        }
    }

    pub fn parse(name: &str) -> Option<Layout> {
        Layout::ALL
            .into_iter()
            .find(|layout| layout.as_str() == name)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RepresentationInfo {
    pub id: RepresentationId,
    /// Unique storage name: the representation name for packet storage
    /// (`q4k`), `representation@layout` otherwise (`q4k@rows16`).
    pub name: &'static str,
    /// The logical representation this storage encodes (`q4k`).
    pub representation: &'static str,
    pub layout: Layout,
    pub kind: RepresentationKind,
    /// Dtype produced by reading one element.
    pub decoded: DType,
    /// Whether the registry defines canonical logical writes/updates.
    pub access: RepresentationAccess,
    /// How a matrix product over this representation takes its activation
    /// operands.
    pub operands: OperandRecipe,
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
    /// A packed representation in the `packet` layout.
    Packed(PackedPacketLayout),
    /// A packed representation in a row layout (`rows16`, `mma16`).
    PackedRows(PackedRowLayout),
    /// Canonical bytes supplied by an external artifact. External packets are
    /// never element-readable or writable; a registered exact conversion is
    /// the sole transition into resident storage.
    External(ExternalPacketLayout),
}

impl RepresentationKind {
    pub const fn form(&self) -> RepresentationForm {
        match self {
            Self::Dense(_) => RepresentationForm::Dense,
            Self::PackedRows(_) => RepresentationForm::Rows,
            Self::Packed(_) => RepresentationForm::Packet,
            Self::External(_) => RepresentationForm::External,
        }
    }
}

/// A representation's storage form: its kind without its layout.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum RepresentationForm {
    Dense,
    Rows,
    Packet,
    External,
}

impl RepresentationForm {
    pub const ALL: [Self; 4] = [Self::Dense, Self::Rows, Self::Packet, Self::External];

    pub const fn name(self) -> &'static str {
        match self {
            Self::Dense => "DENSE",
            Self::Rows => "ROWS",
            Self::Packet => "PACKET",
            Self::External => "EXTERNAL",
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ExternalPacketLayout {
    pub packet_size: u32,
    /// The alignment every packet of a stored tensor has, which a tensor of
    /// the representation starts at (allocations, and views at offsets,
    /// are checked against it): kernels read a source packet in units of
    /// at most this many bytes.
    pub packet_alignment: u32,
    pub logical_group: u32,
    pub packing_axis: PackingAxis,
}

/// One registered exact conversion from an external source into resident
/// storage `destination` = (representation, layout). Conversions are keyed by
/// (external source, representation, layout).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RepresentationConversion {
    pub id: RepresentationConversionId,
    pub source: RepresentationId,
    pub destination: RepresentationId,
    /// The source's packets (its `RepresentationKind::External`).
    pub source_layout: &'static ExternalPacketLayout,
    /// The destination's layout.
    pub destination_layout: ConversionLayout,
    /// Source packet -> the destination representation's packet form.
    pub recipe: PacketRepackRecipe,
}

/// The layout of a conversion's destination: its `RepresentationKind`,
/// which is packed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ConversionLayout {
    Packet(&'static PackedPacketLayout),
    Rows(&'static PackedRowLayout),
}

impl ConversionLayout {
    /// The destination's packet form.
    pub fn packet(self) -> &'static PackedPacketLayout {
        match self {
            Self::Packet(packet) => packet,
            Self::Rows(rows) => &rows.packet,
        }
    }
}

impl RepresentationConversion {
    /// Brings `packets`, whole source packets of arbitrary bytes, into the
    /// conversion's domain, changing as few bits as that takes. The domain
    /// is the source packets whose binary16 fields the recipe converts as
    /// numbers ([`PacketRepackRecipe::converted_f16_fields`]) are finite: a
    /// NaN's payload is not part of the contract, since a device's
    /// conversion quiets a signaling NaN where the host reference widens its
    /// bits. Each such field has the high bit of its exponent cleared.
    pub fn admit_source(&self, packets: &mut [u8]) {
        let fields = self.recipe.converted_f16_fields();
        for packet in packets.chunks_exact_mut(self.source_layout.packet_size as usize) {
            for field in &fields {
                let bit = (field + 14) as usize;
                packet[bit / 8] &= !(1 << (bit % 8));
            }
        }
    }
}

/// `count` deterministic pseudo-random bytes from `seed` (xorshift):
/// arbitrary source bytes, which [`RepresentationConversion::admit_source`]
/// brings into a conversion's domain. For tests (`fixtures`).
#[cfg(any(test, feature = "fixtures"))]
pub fn pseudo_random_bytes(seed: u64, count: usize) -> Vec<u8> {
    let mut state = 0x9e37_79b9_7f4a_7c15u64 ^ seed;
    (0..count)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state >> 24) as u8
        })
        .collect()
}

/// Exact, validated ownership for one external packet -> resident packet-form
/// conversion. There is exactly one recipe for every packet-form plane and
/// each recipe completely initializes that plane.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PacketRepackRecipe {
    pub planes: Vec<PlaneRepackRecipe>,
}

impl PacketRepackRecipe {
    /// The first bit of every binary16 field of the source packet the recipe
    /// converts as a number (the operand of an `F16ToF32`), ascending.
    pub fn converted_f16_fields(&self) -> Vec<u32> {
        fn collect(expression: &RepackExpr, fields: &mut Vec<u32>) {
            match expression {
                RepackExpr::SourceBits { .. } => {}
                RepackExpr::F16ToF32(value) => match &**value {
                    RepackExpr::SourceBits { bit, .. } => fields.push(*bit),
                    _ => unreachable!("a validated recipe widens a source field"),
                },
                RepackExpr::ShiftLeft { value, .. } | RepackExpr::ScaleF32 { value, .. } => {
                    collect(value, fields)
                }
                RepackExpr::Lookup { index, .. } => collect(index, fields),
                RepackExpr::BitOr(left, right) => {
                    collect(left, fields);
                    collect(right, fields);
                }
            }
        }
        let mut fields = Vec::new();
        for plane in &self.planes {
            match plane {
                PlaneRepackRecipe::BitRoutes(_) => {}
                PlaneRepackRecipe::DenseValues(values)
                | PlaneRepackRecipe::PackedEntries(values) => {
                    for value in values {
                        collect(value, &mut fields);
                    }
                }
            }
        }
        fields.sort_unstable();
        fields.dedup();
        fields
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PlaneRepackRecipe {
    /// One source packet bit ordinal for every destination plane bit.
    BitRoutes(Vec<u32>),
    /// One typed expression for every dense destination plane element.
    DenseValues(Vec<RepackExpr>),
    /// One integer expression for every entry of a packed destination plane,
    /// in entry order. Each entry stores the low `entry_bits` bits of its
    /// value (two's complement for a negative value); the expression's range
    /// fits the entry, so nothing is truncated.
    PackedEntries(Vec<RepackExpr>),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RepackExpr {
    SourceBits {
        bit: u32,
        width: u8,
    },
    ShiftLeft {
        value: Box<RepackExpr>,
        bits: u8,
    },
    BitOr(Box<RepackExpr>, Box<RepackExpr>),
    F16ToF32(Box<RepackExpr>),
    /// `value * 2^exponent`: exact for a widened binary16 `value`, whose
    /// every multiple by a small power of two is a normal `f32`.
    ScaleF32 {
        value: Box<RepackExpr>,
        exponent: i8,
    },
    /// `table[index]` of a registered codebook; `index` is an integer
    /// expression whose range lies within the table.
    Lookup {
        index: Box<RepackExpr>,
        table: RepackTable,
    },
}

/// The packet form of a packed representation: its planes interleaved within
/// one packet of `group` logical values. It is the byte geometry of the
/// `packet` layout and the plane schema every layout of the representation
/// places (`PackedRowLayout::packet`). Every backend, planner and runtime
/// consumes this descriptor; none reconstructs packed geometry from
/// representation names or decoder implementation details.
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
            || !self.planes.iter().all(|plane| {
                planes
                    .iter()
                    .filter(|(name, _)| *name == plane.name)
                    .count()
                    == 1
            })
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
            usize::try_from(total)
                .expect("canonical packet bytes fit the host address space")
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

/// Row layouts are consumed by native implementations, the host oracle and
/// host transfers. Compiled construction admits only `packet` storage: the
/// runtime refuses a row-layout element binding before construction starts,
/// so a row layout inside compiled construction is a contradiction.
pub const ROW_LAYOUT_IS_NATIVE_ONLY: &str =
    "row-layout storage reached compiled construction, which admits only `packet` storage";

/// Byte alignment of every row, and of every plane within a row, of a row
/// layout.
pub const ROW_ALIGNMENT: u64 = 16;
/// Rows per `mma16` tile: the `m` extent of `mma.sync.m16n8k16`.
pub const MMA_TILE_ROWS: u64 = 16;
/// Columns per `mma16` k-block: four `k16` fragment steps.
pub const MMA_KBLOCK: u64 = 64;
/// Rows per `rows32` tile: the lanes of a 32-lane cooperative multiply.
pub const ROWS32_TILE_ROWS: u64 = 32;
/// Columns per `rows32` code chunk: one packet of 32 codes of one row.
pub const ROWS32_CHUNK: u64 = 32;
/// Lanes of the warp that owns one `mma16` tile.
pub const MMA_LANES: u64 = 32;

/// The byte geometry of a packed representation in a row layout (`rows16`,
/// `mma16`). A tensor `[..., N, K]` is stored as rows of `K` logical values
/// (every leading coordinate is a row). Within a row the planes follow one
/// another in `planes` order, each starting 16 B-aligned; the row stride is
/// the padded sum, so every row starts 16 B-aligned. A row holds
/// `row_groups(K)` whole storage groups; bytes of padding groups are zero.
///
/// `mma16` additionally pads the row axis `N` of every `[N, K]` matrix to a
/// multiple of 16 rows (padding rows are zero) and permutes each code plane
/// within each 16-row tile (`code_bit`). Scale and super planes, the row
/// stride and every plane offset are exactly those of `rows16`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PackedRowLayout {
    /// `Rows16`, `Rows8` or `Mma16`.
    pub layout: Layout,
    /// The representation's packet form: logical group, plane schema and the
    /// packet every conversion produces before placement.
    pub packet: PackedPacketLayout,
    /// Row planes in storage order.
    pub planes: Vec<RowPlaneInfo>,
    /// Storage groups per row round up to a multiple of this (`mma16`
    /// stores whole 64-column k-blocks, and whole 16-byte chunks of every
    /// code plane).
    pub group_multiple: u32,
}

/// A part of a storage group's record in a `records` tile
/// (`PackedRowLayout::record_byte`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Record {
    Entries,
    Factors,
}

/// Byte geometry of the rows of one row-layout tensor
/// (`PackedRowLayout::geometry`): the row stride, and per row plane its
/// offset within a row and its payload bytes.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RowGeometry {
    pub stride: u64,
    pub offsets: Vec<u64>,
    pub bytes_per_row: Vec<u64>,
}

/// One plane of a row layout.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RowPlaneInfo {
    /// `codes_lo`, `codes_hi`, `codes`, `scales` or `supers`.
    pub name: &'static str,
    /// Bytes per storage group (packet) of logical values of one row.
    pub bytes_per_group: u32,
    pub content: RowPlaneContent,
}

/// What a row plane holds, in terms of the representation's packet form.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RowPlaneContent {
    /// Bits `[shift, shift + bits)` of every code of the packet `words`
    /// plane (packet plane 0). In `rows16`, the code of column `c` sits at bit
    /// `c * bits` of the row's plane (little-endian within bytes: code `2i`
    /// in the low nibble of byte `i` for 4-bit codes).
    Codes { shift: u32, bits: u32 },
    /// The whole vector-code entries of the packet `words` plane (one per
    /// `GRID_VALUES` columns), placed as a code plane of `bits` bits a column
    /// (`code_bit`): an entry is the codes of its columns. With `carried`,
    /// the entries also carry that packet plane, the representation's
    /// coefficient codes: bit `j` of a group's code at the grid's
    /// `carried_bit` of the group's entry `j`.
    Grids { bits: u32, carried: Option<u32> },
    /// Bits `[shift, shift + bits)` of every vector-code entry of the packet
    /// `words` plane, the part of an entry below a byte (`grids` holds its
    /// whole bytes), `bits` an entry in entry order. Its share of a storage
    /// group is placed as a `Groups` plane's, its bytes together, except in
    /// `mma16`, which places it as its entries (`code_bit`): a tile's
    /// k-block's are its rows' `8 bits` bits, row by row, so the GEMM stages
    /// them with the entries.
    GridHigh { shift: u32, bits: u32 },
    /// The bytes of the listed packet planes of one packet, concatenated in
    /// list order, per storage group: the local or the super coefficients.
    Groups { level: CoefficientLevel, planes: Vec<u32> },
    /// `mma16` of a carried grid representation: the tile's whole storage
    /// groups as records of its 16 rows (`record_byte`), every group's
    /// carried entries in GEMV lane order, then every group's `factors`
    /// packet planes. The only plane of its layout.
    Records {
        bits: u32,
        carried: u32,
        factors: Vec<u32>,
    },
}

/// The coefficients a `Groups` row plane holds: the local coefficient codes
/// (`scales`) or the super coefficients (`supers`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CoefficientLevel {
    Local,
    Super,
}

impl CoefficientLevel {
    /// The name of the row plane holding this level.
    pub const fn plane(self) -> &'static str {
        match self {
            Self::Local => "scales",
            Self::Super => "supers",
        }
    }
}

/// What a row plane is to a decoder: the codes (low code plane, grid entries
/// or records), their high plane, the local coefficient codes, or the super
/// coefficients.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RowPlaneRole {
    Codes,
    High,
    Scales,
    Supers,
}

impl RowPlaneRole {
    pub const ALL: [Self; 4] = [Self::Codes, Self::High, Self::Scales, Self::Supers];
    /// The roles of the planes holding codes: those of
    /// [`RowPlaneContent::Codes`], `Grids` and `Records`
    /// ([`RowPlaneInfo::role`]).
    pub const CODES: [Self; 2] = [Self::Codes, Self::High];

    pub const fn name(self) -> &'static str {
        match self {
            Self::Codes => "codes",
            Self::High => "high",
            Self::Scales => CoefficientLevel::Local.plane(),
            Self::Supers => CoefficientLevel::Super.plane(),
        }
    }
}

impl RowPlaneInfo {
    pub fn role(&self) -> RowPlaneRole {
        match self.content {
            RowPlaneContent::Codes { shift: 0, .. }
            | RowPlaneContent::Grids { .. }
            | RowPlaneContent::Records { .. } => RowPlaneRole::Codes,
            RowPlaneContent::Codes { .. } | RowPlaneContent::GridHigh { .. } => RowPlaneRole::High,
            RowPlaneContent::Groups { level: CoefficientLevel::Local, .. } => RowPlaneRole::Scales,
            RowPlaneContent::Groups { level: CoefficientLevel::Super, .. } => RowPlaneRole::Supers,
        }
    }
}

impl PackedRowLayout {
    /// Logical values per storage group.
    pub fn group(&self) -> u32 {
        self.packet.group
    }

    /// Rows of one row tile, or 1.
    pub fn tile_rows(&self) -> u64 {
        match self.layout {
            Layout::Mma16 => MMA_TILE_ROWS,
            Layout::Rows32 => ROWS32_TILE_ROWS,
            Layout::Rows8 => 8,
            Layout::Rows16 => 1,
            Layout::Packet => unreachable!("a row layout is never `packet`"),
        }
    }

    /// Stored storage groups of a row of `logical_extent` values.
    pub fn row_groups(&self, logical_extent: u64) -> Option<u64> {
        let multiple = u64::from(self.group_multiple);
        logical_extent
            .div_ceil(u64::from(self.packet.group))
            .div_ceil(multiple)
            .checked_mul(multiple)
    }

    /// Payload bytes of plane `plane` in one row (excluding alignment
    /// padding).
    pub fn plane_bytes_per_row(&self, plane: usize, logical_extent: u64) -> Option<u64> {
        self.row_groups(logical_extent)?
            .checked_mul(u64::from(self.planes.get(plane)?.bytes_per_group))
    }

    /// Byte offset of plane `plane` within a row.
    pub fn plane_row_offset(&self, plane: usize, logical_extent: u64) -> Option<u64> {
        let mut offset = 0u64;
        for index in 0..plane {
            offset = align(offset.checked_add(self.plane_bytes_per_row(index, logical_extent)?)?)?;
        }
        (plane < self.planes.len()).then_some(offset)
    }

    /// The byte geometry of rows of `logical_extent` values.
    pub fn geometry(&self, logical_extent: u64) -> Option<RowGeometry> {
        Some(RowGeometry {
            stride: self.row_stride_bytes(logical_extent)?,
            offsets: (0..self.planes.len())
                .map(|plane| self.plane_row_offset(plane, logical_extent))
                .collect::<Option<_>>()?,
            bytes_per_row: (0..self.planes.len())
                .map(|plane| self.plane_bytes_per_row(plane, logical_extent))
                .collect::<Option<_>>()?,
        })
    }

    /// Bytes between the starts of adjacent rows.
    pub fn row_stride_bytes(&self, logical_extent: u64) -> Option<u64> {
        let last = self.planes.len().checked_sub(1)?;
        align(
            self.plane_row_offset(last, logical_extent)?
                .checked_add(self.plane_bytes_per_row(last, logical_extent)?)?,
        )
    }

    /// Storage units of a tensor with `extents`: the leading extents (the
    /// row axis padded to whole tiles) and one unit, a row, for the packing
    /// axis. `None` below the rank the layout requires (1, or 2 for `mma16`)
    /// or on overflow.
    pub fn storage_units(&self, extents: &[u64]) -> Option<Vec<u64>> {
        let (_, leading) = extents.split_last()?;
        let mut units = leading.to_vec();
        if self.tile_rows() > 1 {
            let rows = units.last_mut()?;
            let tile = self.tile_rows();
            *rows = rows.div_ceil(tile).checked_mul(tile)?;
        }
        units.push(1);
        Some(units)
    }

    /// Canonical bytes of a tensor with `extents`.
    pub fn bytes(&self, extents: &[u64]) -> Option<u64> {
        let rows = self
            .storage_units(extents)?
            .iter()
            .try_fold(1u64, |rows, extent| rows.checked_mul(*extent))?;
        rows.checked_mul(self.row_stride_bytes(*extents.last()?)?)
    }

    /// The stored row holding logical row `row` (row-major over the leading
    /// extents). The caller established `extents` has a canonical byte count.
    pub fn stored_row(&self, extents: &[u64], row: u64) -> u64 {
        match self.layout {
            Layout::Mma16 | Layout::Rows8 | Layout::Rows32 => {
                let rows = extents[extents.len() - 2];
                let tile = self.tile_rows();
                row / rows * rows.div_ceil(tile) * tile + row % rows
            }
            Layout::Rows16 => row,
            Layout::Packet => unreachable!("a row layout is never `packet`"),
        }
    }

    /// Absolute bit of the first bit of the code of column `column` of stored
    /// row `row` in code plane `plane`, for rows of `logical_extent` values.
    ///
    /// `rows16`: bit `column * bits` of the row's plane.
    /// `rows8`: corresponding storage groups of eight rows are adjacent.
    ///
    /// `mma16`: for tile `T` (stored rows `16T..16T+15`), let `V` be the
    /// concatenation, in row order, of the 16 rows' plane payloads
    /// (`plane_bytes_per_row` bytes each; alignment padding excluded). `V`
    /// is laid out as `[k-block kb][lane l][4 * bits bytes]`, where a k-block
    /// covers 64 columns and lane `l = 4g + t` (`g, t` as in the PTX
    /// m16n8k16 A-fragment). Within a lane's `4 * bits` bytes, k16 step `s`
    /// (columns `64kb + 16s ..+16`) occupies bits `[8 * bits * s, +8 * bits)`,
    /// and slot `j` of that step holds its code at bit `bits * j`. The slots
    /// are the fragment elements in the order `[a0, a2, a4, a6, a1, a3, a5,
    /// a7]`, where `a0, a1` = (row `g`, k `2t, 2t+1`), `a2, a3` = (row `g+8`,
    /// k `2t, 2t+1`), `a4..a7` = the same at k `2t+8, 2t+9`. Hence
    /// `(w >> 4i) & 0x000f000f` of a lane's 32-bit step word of 4-bit codes
    /// is the f16x2 code pair of A register `i`. A `grids` plane's entries
    /// cannot be split across lanes: its `V` is `[k-block kb][row r][64 *
    /// bits / 8 bytes]`, the k-block's columns of each of the tile's rows
    /// adjacent, row by row. A `records` plane's entries are placed by
    /// `record_byte`.
    pub fn code_bit(&self, geometry: &RowGeometry, plane: usize, row: u64, column: u64) -> u64 {
        // `mma16` places a grid entry's high part as it places the entry
        // (`grids`), `bits` an entry: its columns share it.
        let (bits, grids, per) = match self.planes[plane].content {
            RowPlaneContent::Codes { bits, .. } => (bits, false, 1),
            RowPlaneContent::Grids { bits, .. } => (bits, true, 1),
            RowPlaneContent::GridHigh { bits, .. } if self.layout == Layout::Mma16 => {
                (bits, true, u64::from(GRID_VALUES))
            }
            RowPlaneContent::Records { bits, .. } => {
                let entry = column % u64::from(self.packet.group) / u64::from(GRID_VALUES);
                let packet = column / u64::from(self.packet.group);
                let byte = self.record_byte(geometry, row, packet, Record::Entries, |r| {
                    let (q, e) = (entry / 8, entry % 8);
                    q / 2 * 512 + (4 * (r % 8) + e % 4) * 16 + (q % 2 * 2 + r / 8) * 4 + e / 4 * 2
                });
                return byte * 8 + column % u64::from(GRID_VALUES) * u64::from(bits);
            }
            RowPlaneContent::Groups { .. } | RowPlaneContent::GridHigh { .. } => {
                panic!(
                    "row plane `{}` holds no column codes",
                    self.planes[plane].name
                )
            }
        };
        let bits = u64::from(bits);
        let stride = geometry.stride;
        let offset = geometry.offsets[plane];
        match self.layout {
            Layout::Rows16 | Layout::Rows8 => {
                let group = u64::from(self.packet.group);
                let byte = self
                    .share_byte(geometry, row, column / group, plane)
                    .expect("a rows16 or rows8 share is its bytes");
                byte * 8 + column % group * bits
            }
            Layout::Rows32 => interleaved(
                ROWS32_TILE_ROWS,
                ROWS32_CHUNK,
                stride,
                offset,
                row,
                column,
                bits,
            ),
            Layout::Mma16 => {
                let row_bytes = geometry.bytes_per_row[plane];
                let (tile, r) = (row / MMA_TILE_ROWS, row % MMA_TILE_ROWS);
                let (block, c) = (column / MMA_KBLOCK, column % MMA_KBLOCK);
                let bit = if grids {
                    ((block * MMA_TILE_ROWS + r) * (MMA_KBLOCK / per) + c / per) * bits
                } else {
                    let (step, k) = (c / 16, c % 16);
                    let lane = 4 * (r % 8) + (k % 8) / 2;
                    let slot = (k / 8) * 2 + r / 8 + 4 * (k % 2);
                    (block * MMA_LANES + lane) * 32 * bits + 8 * bits * step + bits * slot
                };
                let (byte, within) = (bit / 8, bit % 8);
                let stored = tile * MMA_TILE_ROWS + byte / row_bytes;
                (stored * stride + offset + byte % row_bytes) * 8 + within
            }
            Layout::Packet => unreachable!("a row layout is never `packet`"),
        }
    }

    /// The first byte of storage group `packet`'s share of row plane
    /// `plane` of stored row `row` (its `bytes_per_group` bytes, one after
    /// another), in the layouts that keep a share's bytes together: every
    /// plane of `rows16` and `rows8` (whose tile holds a group's shares of
    /// its rows one after another), and the group planes of `rows32` and
    /// `mma16`, and a grid high plane of `rows32`. `None` for a code plane of
    /// `rows32` or `mma16` and a grid high plane of `mma16` (`code_bit`).
    pub fn share_byte(
        &self,
        geometry: &RowGeometry,
        row: u64,
        packet: u64,
        plane: usize,
    ) -> Option<u64> {
        let group_bytes = u64::from(self.planes[plane].bytes_per_group);
        let (stride, offset, tile) = (geometry.stride, geometry.offsets[plane], self.tile_rows());
        // A grid high plane is placed as a group plane's, but in `mma16` as
        // its entries are (`code_bit`).
        let groups = match self.planes[plane].content {
            RowPlaneContent::Groups { .. } => true,
            RowPlaneContent::GridHigh { .. } => self.layout != Layout::Mma16,
            RowPlaneContent::Codes { .. }
            | RowPlaneContent::Grids { .. }
            | RowPlaneContent::Records { .. } => false,
        };
        Some(match self.layout {
            Layout::Rows16 | Layout::Rows8 => {
                row / tile * stride * tile
                    + offset * tile
                    + packet * group_bytes * tile
                    + row % tile * group_bytes
            }
            // A tile's group plane is `[row][plane bytes of a row]`: every
            // row's groups are contiguous, and the tile's rows follow one
            // another.
            Layout::Rows32 if groups => {
                row / tile * stride * tile
                    + offset * tile
                    + row % tile * geometry.bytes_per_row[plane]
                    + packet * group_bytes
            }
            Layout::Mma16 if groups => row * stride + offset + packet * group_bytes,
            Layout::Rows32 | Layout::Mma16 => return None,
            Layout::Packet => unreachable!("a row layout is never `packet`"),
        })
    }

    /// Bytes of one storage group's entries in a `records` tile: its 16 rows'
    /// entries (the packet `words` plane). The tile's factors start past
    /// every stored group's entries (`record_byte`).
    pub fn record_entry_bytes(&self) -> u64 {
        MMA_TILE_ROWS * u64::from(self.packet.planes[0].bytes_per_group)
    }

    /// The bytes of a `records` plane (`mma16`, its layout's only plane) of
    /// the tile of stored row `row`, `r` the row within the tile: entry byte
    /// `within(r)` of storage group `packet` (`Record::Entries`), or factor
    /// byte `within(r)` of it (`Record::Factors`). A tile's `V` is its 16
    /// rows' bytes whole (alignment padding included: its parts may straddle
    /// rows, whose bytes follow one another), laid out as `[storage group
    /// p][entries of record_entry_bytes]` for every stored group, then
    /// `[storage group p][16 rows' factors]`, so each group's entries are
    /// 1024 bytes on 128-byte lines of their own (`residency.md`). The entries are
    /// in GEMV lane order, `[half h][lane l = 4g + t][16 bytes]`: word `w` of
    /// a lane's 16 bytes is row `g + 8 (w % 2)` of k-block `2h + w / 2`, its
    /// entries `t` (low half) and `4 + t` (high half). A group's factors are
    /// `[g][row g, row g + 8]`: one 32-bit load of a lane gives both of its
    /// rows' f16 factors.
    fn record_byte(
        &self,
        geometry: &RowGeometry,
        row: u64,
        packet: u64,
        part: Record,
        within: impl Fn(u64) -> u64,
    ) -> u64 {
        let (tile, r) = (row / MMA_TILE_ROWS, row % MMA_TILE_ROWS);
        let entries = self.record_entry_bytes();
        let tile_base = tile * MMA_TILE_ROWS * geometry.stride;
        match part {
            Record::Entries => tile_base + packet * entries + within(r),
            Record::Factors => {
                let groups = geometry.bytes_per_row[0] / u64::from(self.planes[0].bytes_per_group);
                let factors = MMA_TILE_ROWS * u64::from(self.planes[0].bytes_per_group) - entries;
                tile_base + groups * entries + packet * factors + within(r)
            }
        }
    }

    /// The row plane holding bit `bit` of packet-form plane `plane`, and the
    /// bit of one storage group's share of that plane holding it. A code
    /// plane's share is the group's columns in order, `bits` each (as a
    /// `rows16` row holds them): plane 0's codes sliced into code planes, or
    /// its vector-code entries kept whole in a grids or records plane (an
    /// entry is its columns' bits), and a carried plane's bit `j` of field
    /// `f` at the carried bit of entry `j` of the field's group. A groups
    /// plane's share is its listed packet planes' bytes in list order; a
    /// records plane's share is its entries, then its factors' bytes in list
    /// order.
    pub fn group_bit(&self, plane: usize, bit: u32) -> (usize, u32) {
        let entries = self.planes.iter().position(|candidate| {
            matches!(
                candidate.content,
                RowPlaneContent::Grids { .. } | RowPlaneContent::Records { .. }
            )
        });
        if plane == 0 {
            let code_bits = self.packet.planes[0].entry_bits;
            let (code, code_bit) = (bit / code_bits, bit % code_bits);
            if let Some(index) = entries {
                return self.entry_bit(index, code, code_bit);
            }
            return self
                .planes
                .iter()
                .enumerate()
                .find_map(|(index, candidate)| match candidate.content {
                    RowPlaneContent::Codes { shift, bits }
                        if (shift..shift + bits).contains(&code_bit) =>
                    {
                        Some((index, code * bits + code_bit - shift))
                    }
                    _ => None,
                })
                .expect("every code bit has one row code plane");
        }
        // The bytes of `planes` before `plane`'s, when it is one of them.
        let prefix = |planes: &[u32]| {
            let position = planes.iter().position(|member| *member as usize == plane)?;
            Some(
                planes[..position]
                    .iter()
                    .map(|member| self.packet.planes[*member as usize].bytes_per_group)
                    .sum::<u32>(),
            )
        };
        if let Some(index) = entries {
            let (carried, factors) = match &self.planes[index].content {
                RowPlaneContent::Grids { carried, .. } => (*carried, None),
                RowPlaneContent::Records {
                    carried, factors, ..
                } => (Some(*carried), Some(factors)),
                _ => unreachable!(),
            };
            let bits = self.column_bits(index);
            if carried == Some(plane as u32) {
                let codes = &self.packet.planes[plane];
                let (field, j) = (bit / codes.entry_bits, bit % codes.entry_bits);
                let entry = field * (codes.group / GRID_VALUES) + j;
                return self.entry_bit(index, entry, self.grid().carried_bit());
            }
            if let Some(factors) = factors {
                let prefix = prefix(factors)
                    .expect("every packet plane of a records layout is in its record");
                return (index, self.packet.group * bits + prefix * 8 + bit);
            }
        }
        self.planes
            .iter()
            .enumerate()
            .find_map(|(index, candidate)| match &candidate.content {
                RowPlaneContent::Groups { planes, .. } => Some((index, prefix(planes)? * 8 + bit)),
                RowPlaneContent::Codes { .. }
                | RowPlaneContent::Grids { .. }
                | RowPlaneContent::GridHigh { .. }
                | RowPlaneContent::Records { .. } => None,
            })
            .expect("every packet plane has one row plane")
    }

    /// The row plane holding bit `bit` of vector-code entry `entry` of a
    /// storage group, and its share bit: the entries plane `entries` holds
    /// an entry's whole bytes (`bits` a column), the `GridHigh` plane the
    /// rest.
    fn entry_bit(&self, entries: usize, entry: u32, bit: u32) -> (usize, u32) {
        let low = GRID_VALUES * self.column_bits(entries);
        if bit < low {
            return (entries, entry * low + bit);
        }
        self.planes
            .iter()
            .enumerate()
            .find_map(|(index, candidate)| match candidate.content {
                RowPlaneContent::GridHigh { shift, bits }
                    if (shift..shift + bits).contains(&bit) =>
                {
                    Some((index, entry * bits + bit - shift))
                }
                _ => None,
            })
            .expect("every entry bit has one row plane")
    }

    /// The bits `[first, first + bits)` of a packet code (a code plane's
    /// codes, a vector-code entry) that row plane `plane` holds, `None` for a
    /// coefficient plane.
    pub fn code_run(&self, plane: usize) -> Option<(u32, u32)> {
        match self.planes[plane].content {
            RowPlaneContent::Codes { shift, bits } | RowPlaneContent::GridHigh { shift, bits } => {
                Some((shift, bits))
            }
            RowPlaneContent::Grids { bits, .. } | RowPlaneContent::Records { bits, .. } => {
                Some((0, GRID_VALUES * bits))
            }
            RowPlaneContent::Groups { .. } => None,
        }
    }

    /// Bits a column of code plane `plane` holds.
    pub(crate) fn column_bits(&self, plane: usize) -> u32 {
        match self.planes[plane].content {
            RowPlaneContent::Codes { bits, .. }
            | RowPlaneContent::Grids { bits, .. }
            | RowPlaneContent::Records { bits, .. } => bits,
            RowPlaneContent::Groups { .. } | RowPlaneContent::GridHigh { .. } => {
                panic!(
                    "row plane `{}` holds no column codes",
                    self.planes[plane].name
                )
            }
        }
    }

    /// Absolute bit holding bit `bit` of packet-form plane `plane` of packet
    /// (storage group) `packet` of stored row `row`: its storage group's
    /// share bit (`group_bit`) placed by the layout.
    pub fn packet_bit(
        &self,
        geometry: &RowGeometry,
        row: u64,
        packet: u64,
        plane: usize,
        bit: u32,
    ) -> u64 {
        let (index, share) = self.group_bit(plane, bit);
        // A code plane's share bit: its column's code bit.
        let code = |bits: u32| {
            let column = packet * u64::from(self.packet.group) + u64::from(share / bits);
            self.code_bit(geometry, index, row, column) + u64::from(share % bits)
        };
        match &self.planes[index].content {
            RowPlaneContent::Codes { bits, .. } | RowPlaneContent::Grids { bits, .. } => {
                code(*bits)
            }
            RowPlaneContent::Records { bits, .. } if share < self.packet.group * bits => {
                code(*bits)
            }
            RowPlaneContent::Records { bits, factors, .. } => {
                let width = factors
                    .iter()
                    .map(|member| u64::from(self.packet.planes[*member as usize].bytes_per_group))
                    .sum::<u64>();
                let within = share - self.packet.group * bits;
                let byte = self.record_byte(geometry, row, packet, Record::Factors, |r| {
                    (2 * (r % 8) + r / 8) * width + u64::from(within / 8)
                });
                byte * 8 + u64::from(within % 8)
            }
            RowPlaneContent::Groups { .. } => {
                let byte = self
                    .share_byte(geometry, row, packet, index)
                    .expect("a group plane's share is its bytes");
                byte * 8 + u64::from(share)
            }
            RowPlaneContent::GridHigh { bits, .. } => {
                match self.share_byte(geometry, row, packet, index) {
                    Some(byte) => byte * 8 + u64::from(share),
                    // Its entry's first column's.
                    None => {
                        let column = packet * u64::from(self.packet.group)
                            + u64::from(share / bits * GRID_VALUES);
                        self.code_bit(geometry, index, row, column) + u64::from(share % bits)
                    }
                }
            }
        }
    }

    /// Whether bit `bit` of packet-form plane `plane` holds information the
    /// layout places. A carried grid representation's packet entries keep
    /// their carried bits zero: the layout's entries hold the coefficient
    /// codes there instead.
    pub fn occupies(&self, plane: usize, bit: u32) -> bool {
        !(plane == 0
            && self
                .carried()
                .is_some_and(|(_, carried_bit)| bit % self.packet.planes[0].entry_bits >= carried_bit))
    }

    /// The packet plane a carried grid layout's entries carry, and the entry
    /// bit carrying it (`GridCodes::carried_bit`).
    pub fn carried(&self) -> Option<(u32, u32)> {
        self.planes.iter().find_map(|candidate| match candidate.content {
            RowPlaneContent::Grids {
                carried: Some(plane),
                ..
            }
            | RowPlaneContent::Records {
                carried: plane, ..
            } => Some((plane, self.grid().carried_bit())),
            _ => None,
        })
    }

    /// The vector codes of a grid representation's `words` plane.
    fn grid(&self) -> &'static GridCodes {
        match &self.packet.planes[0].encoding {
            PlaneEncoding::Packed {
                interpretation: CodeInterpretation::Grid(grid),
                ..
            } => grid,
            _ => panic!("a carried coefficient plane needs grid codes"),
        }
    }

    /// `width` bits of packet-form plane `plane` starting at bit `first`, of
    /// packet `packet` of stored row `row`, from canonical layout bytes.
    #[allow(clippy::too_many_arguments)]
    pub fn read_packet_bits(
        &self,
        bytes: &[u8],
        geometry: &RowGeometry,
        row: u64,
        packet: u64,
        plane: usize,
        first: u32,
        width: u32,
    ) -> u32 {
        (0..width).fold(0, |value, offset| {
            let bit = self.packet_bit(geometry, row, packet, plane, first + offset);
            value | (u32::from((bytes[(bit / 8) as usize] >> (bit % 8)) & 1) << offset)
        })
    }

    /// Place packet-form bytes (`rows × packet_extent(K)` packets of
    /// `packet.packet_size` bytes, ordered by row then packet) of a tensor with
    /// `extents` into this layout's canonical bytes. The caller has
    /// established the canonical byte count of `extents`; a wrong packet byte
    /// count is a caller contradiction.
    pub fn place(&self, extents: &[u64], packets: &[u8]) -> Vec<u8> {
        let mut bytes = vec![
            0u8;
            usize::try_from(
                self.bytes(extents)
                    .expect("row layout tensor has a canonical byte count")
            )
            .expect("row layout bytes fit the host address space")
        ];
        self.visit_packet_bits(extents, packets.len(), |source, destination| {
            let bit = (packets[source / 8] >> (source % 8)) & 1;
            bytes[destination / 8] |= bit << (destination % 8);
        });
        bytes
    }

    /// The packet-form bytes of canonical layout bytes: the inverse of
    /// `place` over logical rows and groups.
    pub fn packets(&self, extents: &[u64], bytes: &[u8]) -> Vec<u8> {
        let (&extent, leading) = extents
            .split_last()
            .expect("row layout tensor has a packing axis");
        let rows: u64 = leading.iter().product();
        let length = rows * self.packet.packet_extent(extent) * u64::from(self.packet.packet_size);
        let mut packets = vec![0u8; length as usize];
        self.visit_packet_bits(extents, packets.len(), |source, destination| {
            let bit = (bytes[destination / 8] >> (destination % 8)) & 1;
            packets[source / 8] |= bit << (source % 8);
        });
        packets
    }

    /// Visit every occupied packet-form bit as (packet-form bit, layout bit).
    fn visit_packet_bits(
        &self,
        extents: &[u64],
        packet_bytes: usize,
        mut visit: impl FnMut(usize, usize),
    ) {
        let (&extent, leading) = extents
            .split_last()
            .expect("row layout tensor has a packing axis");
        let rows: u64 = leading.iter().product();
        let packets_per_row = self.packet.packet_extent(extent);
        let packet_size = u64::from(self.packet.packet_size);
        let geometry = self
            .geometry(extent)
            .expect("row layout tensor has a canonical byte count");
        assert_eq!(
            rows * packets_per_row * packet_size,
            packet_bytes as u64,
            "packet-form byte count differs from the tensor's packet count"
        );
        for row in 0..rows {
            let stored = self.stored_row(extents, row);
            for packet in 0..packets_per_row {
                let base = (row * packets_per_row + packet) * packet_size * 8;
                for (index, plane) in self.packet.planes.iter().enumerate() {
                    for bit in 0..plane.bytes_per_group * 8 {
                        if !self.occupies(index, bit) {
                            continue;
                        }
                        let source = base + u64::from(plane.offset) * 8 + u64::from(bit);
                        let destination = self.packet_bit(&geometry, stored, packet, index, bit);
                        visit(source as usize, destination as usize);
                    }
                }
            }
        }
    }
}

fn align(bytes: u64) -> Option<u64> {
    Some(bytes.checked_add(ROW_ALIGNMENT - 1)? / ROW_ALIGNMENT * ROW_ALIGNMENT)
}

/// Bit of the code of `column` of stored `row` in a code plane (at row
/// offset `offset`, `bits` a column) interleaved across `tile` rows: a
/// tile's plane is `[chunk][row][chunk bytes]`, `chunk` columns of each of
/// its rows adjacent, row by row.
fn interleaved(
    tile: u64,
    chunk: u64,
    stride: u64,
    offset: u64,
    row: u64,
    column: u64,
    bits: u64,
) -> u64 {
    let chunk_bytes = chunk * bits / 8;
    let byte = row / tile * stride * tile
        + offset * tile
        + column / chunk * chunk_bytes * tile
        + row % tile * chunk_bytes;
    byte * 8 + column % chunk * bits
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
/// Every registered conversion, in id order.
pub fn representation_conversions() -> &'static [RepresentationConversion] {
    internals::representation_conversions()
}
/// A GGUF source the registry converts from: its ggml type, its storage
/// name, and the resident representation its conversion writes. A source
/// packet is one block of the type (`magnitude_ggml`'s geometry).
#[derive(Clone, Copy, Debug)]
struct GgufSource {
    encoding: Encoding,
    name: &'static str,
    resident: &'static str,
}

const fn gguf_source_spec(
    encoding: Encoding,
    name: &'static str,
    resident: &'static str,
) -> GgufSource {
    GgufSource {
        encoding,
        name,
        resident,
    }
}

const GGUF_SOURCES: [GgufSource; 19] = [
    gguf_source_spec(Encoding::Q3K, "gguf_q3_k", "q3k"),
    gguf_source_spec(Encoding::Iq3S, "gguf_iq3_s", "iq3s"),
    gguf_source_spec(Encoding::Q4K, "gguf_q4_k", "q4k"),
    gguf_source_spec(Encoding::Q5K, "gguf_q5_k", "q5k"),
    gguf_source_spec(Encoding::Q6K, "gguf_q6_k", "q6k"),
    gguf_source_spec(Encoding::Q8_0, "gguf_q8_0", "q8g32s"),
    gguf_source_spec(Encoding::Iq4Xs, "gguf_iq4_xs", "iq4xs"),
    gguf_source_spec(Encoding::Q4_0, "gguf_q4_0", "q4g32s"),
    gguf_source_spec(Encoding::Q4_1, "gguf_q4_1", "q4g32"),
    gguf_source_spec(Encoding::Iq4Nl, "gguf_iq4_nl", "iq4g32s"),
    gguf_source_spec(Encoding::Q5_0, "gguf_q5_0", "q5g32s"),
    gguf_source_spec(Encoding::Q5_1, "gguf_q5_1", "q5g32"),
    gguf_source_spec(Encoding::Mxfp4, "gguf_mxfp4", "mxfp4g32"),
    gguf_source_spec(Encoding::Nvfp4, "gguf_nvfp4", "nvfp4g16"),
    gguf_source_spec(Encoding::Iq2Xs, "gguf_iq2_xs", "iq2xs"),
    gguf_source_spec(Encoding::Iq3Xxs, "gguf_iq3_xxs", "iq3xxs"),
    gguf_source_spec(Encoding::Iq2Xxs, "gguf_iq2_xxs", "iq2xxs"),
    gguf_source_spec(Encoding::Q2K, "gguf_q2_k", "q2k"),
    gguf_source_spec(Encoding::Iq2S, "gguf_iq2_s", "iq2s"),
];

/// The external storage `encoding` is read as: its registered source;
/// `None` when the registry converts from none.
pub fn gguf_source(encoding: Encoding) -> Option<RepresentationId> {
    GGUF_SOURCES
        .iter()
        .find(|source| source.encoding == encoding)
        .and_then(|source| representation(source.name))
}

/// The storage `source`'s registered conversion writes in `layout` for
/// products taking `operands`. A source converts into one representation,
/// which has one element per layout and operand recipe, so it is unique.
pub fn conversion_destination(
    source: RepresentationId,
    layout: Layout,
    operands: OperandRecipe,
) -> Option<RepresentationId> {
    representations()
        .iter()
        .find(|info| {
            info.layout == layout
                && info.operands == operands
                && representation_conversion(source, info.id).is_some()
        })
        .map(|info| info.id)
}
/// A registry table indexed by one kind of typed registry identity.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Table {
    Capability,
    Intrinsic,
    Representation,
    RepresentationConversion,
}
/// The number of rows of `table`.
pub(crate) fn table_len(table: Table) -> usize {
    internals::table_len(table)
}
/// The registry's own `'static` copy of an intrinsic or opaque-value name it
/// declares, or `None` when the registry declares no such name.
pub(crate) fn declared_name(name: &str) -> Option<&'static str> {
    internals::declared_name(name)
}
/// A name the registry declares. Fields of this type deserialize through
/// [`deserialize_declared_name`]; naming the type keeps serde from treating
/// the field as borrowed from the input.
pub(crate) type DeclaredName = &'static str;
/// Deserializes a registry-declared name back to the registry's own copy.
pub(crate) fn deserialize_declared_name<'de, D: serde::Deserializer<'de>>(
    deserializer: D,
) -> Result<&'static str, D::Error> {
    use serde::de::Error as _;
    let name = <&'de str as serde::Deserialize>::deserialize(deserializer)?;
    declared_name(name).ok_or_else(|| D::Error::custom("name the registry does not declare"))
}
/// The element `storage` (a storage name such as `q4k`, or `nvfp4g16a4` for
/// NVFP4's W4A4 element) in `layout`. Dense storage exists only in `packet`.
pub fn storage(storage: &str, layout: Layout) -> Option<RepresentationId> {
    representation(&stored_name(storage, layout))
}

/// The element whose storage `id` is: `id`'s representation in its layout,
/// multiplying the rounded activation. Storage (its bytes, decode and
/// conversions) is the same for every operand recipe, so `id` itself unless
/// it is a W4A4 element.
pub fn storage_of(id: RepresentationId) -> RepresentationId {
    let info = representation_info(id);
    storage(info.representation, info.layout)
        .expect("every element's representation has its rounded element in its layout")
}

/// The unique name of element `storage` in `layout`: the storage name, then
/// `@layout` in a row layout.
fn stored_name(storage: &str, layout: Layout) -> String {
    match layout {
        Layout::Packet => storage.to_owned(),
        _ => format!("{storage}@{}", layout.as_str()),
    }
}

/// The unique name of representation `representation`'s element of
/// `operands` in `layout`: its storage name is the representation's with
/// the recipe's suffix ([`storage`] finds it).
fn element_name(representation: &str, operands: OperandRecipe, layout: Layout) -> &'static str {
    let storage = format!("{representation}{}", operands.suffix());
    Box::leak(stored_name(&storage, layout).into_boxed_str())
}
/// Canonical storage bytes of a contiguous tensor with `extents` in
/// representation `id`: dense = elements × dtype bytes; packed and external =
/// rows × ceil(last / group) × packet bytes, where rows is the product of the
/// leading extents; row layouts = stored rows × row stride
/// (`PackedRowLayout::bytes`). `None` when `extents` is empty for a packed or
/// external representation (or below rank 2 for `mma16`), or when the count
/// overflows `u64`.
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
            .try_fold(u64::from(dtype.bytes()), |bytes, extent| {
                bytes.checked_mul(*extent)
            }),
        RepresentationKind::Packed(layout) => {
            let (rows, last) = rows_and_last()?;
            layout.bytes(rows, last)
        }
        RepresentationKind::PackedRows(layout) => layout.bytes(extents),
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
/// representation in the `packet` layout (`PlaneInfo::storage_dtype`). The
/// checker resolved both the representation and the plane name (plane views
/// exist only for `packet` storage), so either being unknown panics.
pub fn plane_element_dtype(representation: RepresentationId, plane: &str) -> DType {
    let info = representation_info(representation);
    let RepresentationKind::Packed(layout) = &info.kind else {
        panic!(
            "plane view of the non-packed representation `{}`",
            info.name
        )
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
    // A recipe is a pure function of the process-global registry; derive
    // each one once rather than re-evaluating its reference math per query.
    type Recipes = std::collections::HashMap<(RepresentationId, DType), Option<DecodeRecipe>>;
    static RECIPES: std::sync::OnceLock<std::sync::RwLock<Recipes>> = std::sync::OnceLock::new();
    let recipes = RECIPES.get_or_init(Default::default);
    if let Some(known) = recipes.read().unwrap().get(&(id, output)) {
        return known.clone();
    }
    let recipe =
        internals::packed(id).map(|representation| representation.decode_recipe_to(output));
    recipes
        .write()
        .unwrap()
        .entry((id, output))
        .or_insert(recipe)
        .clone()
}

/// The row planes of a `bits`-bit code: a low plane of the largest power
/// of two that fits (1, 2, 4 or 8 bits) and a high plane of the rest, at
/// most 2 bits (3 = 2 + 1, 5 = 4 + 1, 6 = 4 + 2); `None` for a width no
/// split holds.
pub(crate) fn code_planes(bits: u32) -> Option<(u32, u32)> {
    let low = 1u32 << bits.checked_ilog2()?;
    let high = bits - low;
    (low <= 8 && high <= 2).then_some((low, high))
}

/// How a decoder reads row-layout storage `id` (its code planes are split
/// by `code_planes` and grid entries may carry its coefficient codes, which
/// the packet layout's planes do not); `None` for any other storage.
pub fn decode_axes(id: RepresentationId) -> Option<DecodeAxes> {
    match representation_info(id).kind {
        RepresentationKind::PackedRows(_) => internals::packed(id).map(|representation| representation.decode_axes()),
        RepresentationKind::Packed(_) | RepresentationKind::Dense(_) | RepresentationKind::External(_) => None,
    }
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
    use crate::intrinsics::{RowOperand, RowResult, capability_rows};
    use crate::repr::{Coefficients, PlaneEncoding, REPRS, Repr};
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
        /// Built from the representations once they are static
        /// ([`conversions`]): a conversion holds its endpoints' layouts.
        conversions: OnceLock<Vec<RepresentationConversion>>,
    }

    fn build() -> Tables {
        // Representations: every dense dtype first (ordinal = dtype ordinal),
        // then the packed representations in `REPRS` order.
        let mut representations = Vec::new();
        for dtype in DType::ALL {
            representations.push(RepresentationInfo {
                id: RepresentationId::new(representations.len() as u32),
                name: dtype.name(),
                representation: dtype.name(),
                layout: Layout::Packet,
                kind: RepresentationKind::Dense(dtype),
                decoded: dtype,
                access: RepresentationAccess::ReadWrite,
                operands: OperandRecipe::Rounded,
            });
        }
        for repr in REPRS {
            for &operands in repr.operand_recipes() {
                representations.push(RepresentationInfo {
                    id: RepresentationId::new(representations.len() as u32),
                    name: element_name(repr.name, operands, Layout::Packet),
                    representation: repr.name,
                    layout: Layout::Packet,
                    kind: RepresentationKind::Packed(packed_layout(repr)),
                    decoded: DType::F32,
                    access: RepresentationAccess::ReadOnly,
                    operands,
                });
            }
        }
        // (external source, packet bytes, packet values, resident
        // representation). A source whose every value the resident
        // representation holds exactly converts into it: each moves its
        // codes and coefficient fields bit for bit into a representation of
        // its own.
        let external_specs = GGUF_SOURCES.map(|source| {
            let packet_size = u32::try_from(source.encoding.block_bytes())
                .expect("a GGUF source packet fits u32 bytes");
            let logical_group = u32::try_from(source.encoding.block_elements())
                .expect("a GGUF source packet fits u32 values");
            (source.name, packet_size, logical_group, source.resident)
        });
        // A GGUF packet is declared aligned to the largest power of two, to
        // 4 (the widest unit a kernel reads a source packet in), dividing
        // its size. Device allocations are aligned further; a tensor's data
        // in its file starts at a multiple of `general.alignment`, any power
        // of two, so a view of the file's pages that falls short is not
        // made, and the importer uploads the tensor instead (the engine's
        // `ImportSource::acquire`).
        for (name, packet_size, logical_group, _) in external_specs {
            representations.push(RepresentationInfo {
                id: RepresentationId::new(representations.len() as u32),
                name,
                representation: name,
                layout: Layout::Packet,
                kind: RepresentationKind::External(ExternalPacketLayout {
                    packet_size,
                    packet_alignment: crate::repack::alignment(packet_size).min(4),
                    logical_group,
                    packing_axis: PackingAxis::Last,
                }),
                decoded: DType::F32,
                access: RepresentationAccess::ConversionSource,
                // Never element-readable, so never a product's operand.
                operands: OperandRecipe::Rounded,
            });
        }
        // Row layouts exist for every resident representation of an external
        // source: the resident forms the engine imports.
        let mut residents: Vec<&str> = Vec::new();
        for (_, _, _, resident) in external_specs {
            if !residents.contains(&resident) {
                residents.push(resident);
            }
        }
        for resident in residents {
            let repr = REPRS
                .iter()
                .find(|repr| repr.name == resident)
                .expect("external conversion names a registered packed representation");
            for layout in [Layout::Rows16, Layout::Rows8, Layout::Mma16, Layout::Rows32] {
                for &operands in repr.operand_recipes() {
                    representations.push(RepresentationInfo {
                        id: RepresentationId::new(representations.len() as u32),
                        name: element_name(resident, operands, layout),
                        representation: repr.name,
                        layout,
                        kind: RepresentationKind::PackedRows(row_layout(repr, layout)),
                        decoded: DType::F32,
                        access: RepresentationAccess::ReadOnly,
                        operands,
                    });
                }
            }
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
                        let (_, operand) = row
                            .arguments
                            .get(projection.argument as usize)
                            .expect("result axis selects an actual intrinsic argument");
                        let rank = match operand {
                            RowOperand::Readable(_, rank)
                            | RowOperand::ReadableRepresentation(_, rank) => *rank,
                            RowOperand::Scalar(_) => {
                                panic!("result axis cannot select a scalar argument")
                            }
                        };
                        assert!(
                            projection.axis < rank,
                            "result axis lies within its argument rank"
                        );
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
                            rows.iter()
                                .all(|other| categories(*other) != categories(id)),
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
            conversions: OnceLock::new(),
        }
    }

    /// The registered conversions between `representations`: of each
    /// external source into every element of its resident representation.
    fn conversions(
        representations: &'static [RepresentationInfo],
    ) -> Vec<RepresentationConversion> {
        let mut conversions: Vec<RepresentationConversion> = Vec::new();
        for gguf in GGUF_SOURCES {
            let source = representations
                .iter()
                .find(|representation| representation.name == gguf.name)
                .expect("registered external representation is present");
            let RepresentationKind::External(source_layout) = &source.kind else {
                panic!("representation conversion source is not external storage")
            };
            for destination in representations
                .iter()
                .filter(|representation| representation.representation == gguf.resident)
            {
                let destination_layout = match &destination.kind {
                    RepresentationKind::Packed(packet) => ConversionLayout::Packet(packet),
                    RepresentationKind::PackedRows(rows) => ConversionLayout::Rows(rows),
                    RepresentationKind::Dense(_) | RepresentationKind::External(_) => {
                        panic!("external conversion destination is not packed")
                    }
                };
                let packet = destination_layout.packet();
                let recipe = external_repack_recipe(gguf.name, packet);
                validate_repack_recipe(source_layout, packet, &recipe);
                assert!(
                    conversions.iter().all(|conversion| {
                        conversion.source != source.id || conversion.destination != destination.id
                    }),
                    "external representation `{}` has more than one conversion into `{}`",
                    gguf.name,
                    destination.name
                );
                conversions.push(RepresentationConversion {
                    id: RepresentationConversionId::new(conversions.len() as u32),
                    source: source.id,
                    destination: destination.id,
                    source_layout,
                    destination_layout,
                    recipe,
                });
            }
        }
        conversions
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

    /// The row layout of one packed representation. Planes, in order:
    /// 1. code planes from the packet `words` plane (`code_planes`): a low
    ///    plane of the largest power of two of bits that fits (`codes_lo`, or
    ///    `codes` when it holds the whole 8-bit code) and a high plane of the
    ///    rest, at most 2 bits (`codes_hi`); vector-code entries -> `grids`,
    ///    whole, carrying the `coefficients` plane when the representation's
    ///    coefficients are carried;
    /// 2. `scales`: the hierarchical local coefficients exactly as the packet
    ///    `coefficients` plane packs them (never widened), unless carried;
    /// 3. `supers`: every remaining coefficient plane of one packet
    ///    concatenated per group (k-quant super factors `d, dmin`; a direct
    ///    per-group scale).
    ///
    /// `mma16` of a carried representation is one `records` plane holding
    /// the entries, their carried codes and the supers.
    fn row_layout(repr: &Repr, layout: Layout) -> PackedRowLayout {
        let packet = packed_layout(repr);
        let words = &packet.planes[0];
        assert_eq!(words.name, "words", "packet plane 0 holds the codes");
        let group = packet.group;
        let coefficients = |names: &[&str]| -> Vec<u32> {
            packet
                .planes
                .iter()
                .enumerate()
                .filter(|(_, plane)| names.contains(&plane.name))
                .map(|(index, _)| index as u32)
                .collect()
        };
        let group_bytes = |members: &[u32]| {
            members
                .iter()
                .map(|member| packet.planes[*member as usize].bytes_per_group)
                .sum::<u32>()
        };
        let scales = coefficients(&["coefficients"]);
        let supers = coefficients(&[
            "scale_factor",
            "bias_factor",
            "scale",
            "bias",
            "block_scale",
        ]);
        assert!(
            !supers.is_empty(),
            "`{}` has no super coefficients",
            repr.name
        );
        let carried = match repr.coefficients {
            Coefficients::Hierarchical { carried: true, .. } => {
                let CodeInterpretation::Grid(grid) = &repr.code else {
                    panic!("`{}` carries coefficients outside grid entries", repr.name)
                };
                let [plane] = scales[..] else {
                    panic!("`{}` carries one coefficient plane", repr.name)
                };
                let codes = &packet.planes[plane as usize];
                assert!(
                    codes.fields == 1
                        && codes.entry_bits == codes.group / GRID_VALUES
                        && grid.carried_bit() < grid.entry_bits(),
                    "`{}` carries one coefficient bit an entry, a group's code in its entries",
                    repr.name
                );
                Some(plane)
            }
            _ => None,
        };
        let code_plane = |name, shift, bits: u32| RowPlaneInfo {
            name,
            bytes_per_group: group * bits / 8,
            content: RowPlaneContent::Codes { shift, bits },
        };
        let mut planes = match (&words.encoding, words.entry_bits) {
            // A carried grid representation's `mma16` tile is records.
            (_, bits) if layout == Layout::Mma16 && carried.is_some() => {
                assert!(
                    bits == 16 && group == 256 && repr.group == 32,
                    "`{}` records hold 256 values of 16-bit entries, groups of 32",
                    repr.name
                );
                vec![RowPlaneInfo {
                    name: "records",
                    bytes_per_group: words.bytes_per_group + group_bytes(&supers),
                    content: RowPlaneContent::Records {
                        bits: bits / GRID_VALUES,
                        carried: carried.unwrap(),
                        factors: supers.clone(),
                    },
                }]
            }
            // Vector codes stay whole: one entry is several values' codes,
            // its whole bytes in `grids` and the rest, below a byte, in
            // `grids_hi`.
            (
                PlaneEncoding::Packed {
                    interpretation: CodeInterpretation::Grid(_),
                    ..
                },
                bits,
            ) => {
                let (low, high) = (bits / 8 * 8, bits % 8);
                let entries = group / GRID_VALUES;
                assert!(
                    low > 0 && (entries * high).is_multiple_of(8),
                    "`{}` entries are whole bytes and a high part of whole bytes a group",
                    repr.name
                );
                let mut planes = vec![RowPlaneInfo {
                    name: "grids",
                    bytes_per_group: entries * low / 8,
                    content: RowPlaneContent::Grids {
                        bits: low / GRID_VALUES,
                        carried,
                    },
                }];
                if high > 0 {
                    planes.push(RowPlaneInfo {
                        name: "grids_hi",
                        bytes_per_group: entries * high / 8,
                        content: RowPlaneContent::GridHigh {
                            shift: low,
                            bits: high,
                        },
                    });
                }
                planes
            }
            // A low plane of whole bytes is `codes`.
            (_, bits) => {
                let (low, high) = code_planes(bits)
                    .unwrap_or_else(|| panic!("`{}` has no row layout for {bits}-bit codes", repr.name));
                match (low, high) {
                    (8, 0) => vec![code_plane("codes", 0, 8)],
                    (_, 0) => vec![code_plane("codes_lo", 0, low)],
                    _ => vec![
                        code_plane("codes_lo", 0, low),
                        code_plane("codes_hi", low, high),
                    ],
                }
            }
        };
        if !scales.is_empty() && carried.is_none() {
            planes.push(RowPlaneInfo {
                name: CoefficientLevel::Local.plane(),
                bytes_per_group: group_bytes(&scales),
                content: RowPlaneContent::Groups {
                    level: CoefficientLevel::Local,
                    planes: scales,
                },
            });
        }
        if !matches!(planes[0].content, RowPlaneContent::Records { .. }) {
            planes.push(RowPlaneInfo {
                name: CoefficientLevel::Super.plane(),
                bytes_per_group: group_bytes(&supers),
                content: RowPlaneContent::Groups {
                    level: CoefficientLevel::Super,
                    planes: supers,
                },
            });
        }
        // The row planes together cover every packet plane once.
        let mut covered: Vec<u32> = planes
            .iter()
            .flat_map(|plane| match &plane.content {
                RowPlaneContent::Groups { planes, .. } => planes.clone(),
                RowPlaneContent::Codes { .. }
                | RowPlaneContent::GridHigh { .. }
                | RowPlaneContent::Grids { carried: None, .. } => Vec::new(),
                RowPlaneContent::Grids {
                    carried: Some(carried),
                    ..
                } => vec![*carried],
                RowPlaneContent::Records {
                    carried, factors, ..
                } => std::iter::once(*carried).chain(factors.iter().copied()).collect(),
            })
            .collect();
        if planes.iter().any(|plane| {
            matches!(
                plane.content,
                RowPlaneContent::Codes { .. }
                    | RowPlaneContent::Grids { .. }
                    | RowPlaneContent::Records { .. }
            )
        }) {
            covered.push(0);
        }
        covered.sort_unstable();
        assert!(
            covered.iter().copied().eq(0..packet.planes.len() as u32),
            "row layout of `{}` covers every packet plane exactly once",
            repr.name
        );
        let group_multiple = match layout {
            Layout::Rows16 | Layout::Rows8 => 1,
            Layout::Rows32 => {
                assert!(
                    group.is_multiple_of(ROWS32_CHUNK as u32),
                    "`{}` groups do not hold whole 32-column code chunks",
                    repr.name
                );
                1
            }
            Layout::Mma16 => {
                let block = MMA_KBLOCK as u32;
                assert!(
                    block.is_multiple_of(group) || group.is_multiple_of(block),
                    "`{}` groups do not tile 64-column k-blocks",
                    repr.name
                );
                // Whole 64-column k-blocks, and a whole number of 16-byte
                // chunks in every code plane of a row: the GEMM stages a
                // tile's code planes in 16-byte pieces, which must not
                // straddle two rows (a 32-value group's one-bit high plane
                // is 4 bytes, so its rows hold groups in fours).
                let gcd = |mut a: u32, mut b: u32| {
                    while b != 0 {
                        (a, b) = (b, a % b);
                    }
                    a
                };
                planes
                    .iter()
                    .fold((block / group).max(1), |multiple, plane| {
                        let chunk = 16 / gcd(16, plane.bytes_per_group);
                        match plane.content {
                            // A grid high plane is placed as its entries are
                            // (`code_bit`), its k-block's bytes of a tile
                            // fewer than its rows' chunks.
                            RowPlaneContent::Codes { .. } | RowPlaneContent::GridHigh { .. } => {
                                multiple / gcd(multiple, chunk) * chunk
                            }
                            // A tile's k-block of a grids plane is its 16
                            // rows' k-block bytes, whole chunks; records
                            // run over the tile's rows whole.
                            RowPlaneContent::Grids { .. }
                            | RowPlaneContent::Groups { .. }
                            | RowPlaneContent::Records { .. } => multiple,
                        }
                    })
            }
            Layout::Packet => unreachable!("a row layout is never `packet`"),
        };
        PackedRowLayout {
            layout,
            packet,
            planes,
            group_multiple,
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

    /// The resident E2M1 code of the GGUF nibble at source bit `bit` (code 8,
    /// E2M1 -0, becomes ggml's +0).
    fn e2m1_positive_zero(bit: u32) -> RepackExpr {
        RepackExpr::Lookup {
            index: Box::new(RepackExpr::SourceBits { bit, width: 4 }),
            table: RepackTable::E2m1PositiveZero,
        }
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
            "gguf_q2_k" => {
                // block_q2_K: scales[16] (a 4-bit scale, then a 4-bit
                // minimum, per 16 values) | qs[64] | d (f16) | dmin (f16).
                // Value `p`'s code is at bit 2 ((p % 128) / 32) of
                // `qs[32 (p / 128) + p % 32]`.
                push(plane(
                    "words",
                    PlaneRepackRecipe::BitRoutes(
                        (0..512)
                            .map(|bit| {
                                let position = bit / 2;
                                let source_byte = 16 + (position / 128) * 32 + position % 32;
                                source_byte * 8 + (position % 128 / 32) * 2 + bit % 2
                            })
                            .collect(),
                    ),
                ));
                push(plane(
                    "coefficients",
                    PlaneRepackRecipe::BitRoutes(direct_bits(0, 16)),
                ));
                push(plane(
                    "scale_factor",
                    PlaneRepackRecipe::BitRoutes(direct_bits(80, 2)),
                ));
                push(plane(
                    "bias_factor",
                    PlaneRepackRecipe::BitRoutes(direct_bits(82, 2)),
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
            "gguf_iq2_xs" => {
                // block_iq2_xs: d (f16) | qs[32] (u16: grid index, signs) |
                // scales[8] (two 4-bit sub-block scales each, low first).
                push(plane(
                    "words",
                    PlaneRepackRecipe::BitRoutes(direct_bits(2, 64)),
                ));
                push(plane(
                    "coefficients",
                    PlaneRepackRecipe::BitRoutes(direct_bits(66, 8)),
                ));
                push(plane(
                    "scale_factor",
                    PlaneRepackRecipe::DenseValues(vec![RepackExpr::ScaleF32 {
                        value: Box::new(RepackExpr::F16ToF32(Box::new(RepackExpr::SourceBits {
                            bit: 0,
                            width: 16,
                        }))),
                        exponent: -3,
                    }]),
                ));
            }
            "gguf_iq3_xxs" => {
                // block_iq3_xxs: d (f16) | qs[64] (8-bit grid indices of four
                // values) | eight u32, one per 32 values: four 7-bit sign
                // groups of eight values, then the 4-bit scale. Entry `e`
                // (values 8e..8e+8 of sub-block e / 4) is
                // `qs[8 (e / 4) + 2 (e % 4) .. +2] | signs (e % 4) << 16`.
                push(plane(
                    "words",
                    PlaneRepackRecipe::PackedEntries(
                        (0..32)
                            .map(|entry| {
                                let (block, group) = (entry / 4, entry % 4);
                                RepackExpr::BitOr(
                                    Box::new(RepackExpr::SourceBits {
                                        bit: (2 + 8 * block + 2 * group) * 8,
                                        width: 16,
                                    }),
                                    Box::new(RepackExpr::ShiftLeft {
                                        value: Box::new(RepackExpr::SourceBits {
                                            bit: (66 + 4 * block) * 8 + 7 * group,
                                            width: 7,
                                        }),
                                        bits: 16,
                                    }),
                                )
                            })
                            .collect(),
                    ),
                ));
                push(plane(
                    "coefficients",
                    PlaneRepackRecipe::BitRoutes(
                        (0..32)
                            .map(|bit| (66 + 4 * (bit / 4)) * 8 + 28 + bit % 4)
                            .collect(),
                    ),
                ));
                push(plane(
                    "scale_factor",
                    PlaneRepackRecipe::DenseValues(vec![RepackExpr::ScaleF32 {
                        value: Box::new(RepackExpr::F16ToF32(Box::new(RepackExpr::SourceBits {
                            bit: 0,
                            width: 16,
                        }))),
                        exponent: -2,
                    }]),
                ));
            }
            "gguf_iq2_s" => {
                // block_iq2_s: d (f16) | qs[64]: 32 low grid-index bytes,
                // then 32 sign bytes | qh[8] (the indices' two high bits,
                // four entries a byte, lowest first) | scales[8] (two 4-bit
                // scales of 16 values each, low first). Entry `e` (values
                // 8e..8e+8) is its 10-bit index `qs[e] | (qh[e / 4] >> 2 (e %
                // 4) & 3) << 8`, then its signs `qs[32 + e]`, bit for bit.
                push(plane(
                    "words",
                    PlaneRepackRecipe::BitRoutes(
                        (0..32 * 18)
                            .map(|bit| {
                                let (entry, bit) = (bit / 18, bit % 18);
                                match bit {
                                    0..8 => (2 + entry) * 8 + bit,
                                    8..10 => (66 + entry / 4) * 8 + 2 * (entry % 4) + bit - 8,
                                    _ => (34 + entry) * 8 + bit - 10,
                                }
                            })
                            .collect(),
                    ),
                ));
                push(plane(
                    "coefficients",
                    PlaneRepackRecipe::BitRoutes(direct_bits(74, 8)),
                ));
                // d moves bit for bit; the representation scales it by 2^-3.
                push(plane(
                    "scale_factor",
                    PlaneRepackRecipe::BitRoutes(direct_bits(0, 2)),
                ));
            }
            "gguf_iq2_xxs" => {
                // block_iq2_xxs: d (f16) | eight u32 pairs, one per 32
                // values: four 8-bit grid indices of eight values, then four
                // 7-bit sign groups and the 4-bit scale. Entry `e` (values
                // 8e..8e+8 of sub-block e / 4) is
                // `qs[8 (e / 4) + e % 4] | signs (e % 4) << 8`.
                push(plane(
                    "words",
                    PlaneRepackRecipe::PackedEntries(
                        (0..32)
                            .map(|entry| {
                                let (block, group) = (entry / 4, entry % 4);
                                RepackExpr::BitOr(
                                    Box::new(RepackExpr::SourceBits {
                                        bit: (2 + 8 * block + group) * 8,
                                        width: 8,
                                    }),
                                    Box::new(RepackExpr::ShiftLeft {
                                        value: Box::new(RepackExpr::SourceBits {
                                            bit: (6 + 8 * block) * 8 + 7 * group,
                                            width: 7,
                                        }),
                                        bits: 8,
                                    }),
                                )
                            })
                            .collect(),
                    ),
                ));
                push(plane(
                    "coefficients",
                    PlaneRepackRecipe::BitRoutes(
                        (0..32)
                            .map(|bit| (6 + 8 * (bit / 4)) * 8 + 28 + bit % 4)
                            .collect(),
                    ),
                ));
                // d moves bit for bit; the representation scales it by 2^-3.
                push(plane(
                    "scale_factor",
                    PlaneRepackRecipe::BitRoutes(direct_bits(0, 2)),
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
                // block_iq4_xs: d (f16) | scales_h (u16) | scales_l[4] |
                // qs[128]. Value p of sub-block p / 32 is the low (p % 32 <
                // 16) or high nibble of qs[16 (p / 32) + p % 16]; sub-block
                // g's 6-bit scale code is nibble g % 2 of scales_l[g / 2]
                // under bits 2g, 2g + 1 of scales_h; d moves bit for bit.
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
                push(plane(
                    "coefficients",
                    PlaneRepackRecipe::BitRoutes(
                        (0..48)
                            .map(|bit| {
                                let (group, code_bit) = (bit / 6, bit % 6);
                                if code_bit < 4 {
                                    (4 + group / 2) * 8 + (group % 2) * 4 + code_bit
                                } else {
                                    16 + 2 * group + code_bit - 4
                                }
                            })
                            .collect(),
                    ),
                ));
                push(plane(
                    "scale_factor",
                    PlaneRepackRecipe::BitRoutes(direct_bits(0, 2)),
                ));
            }
            "gguf_q3_k" => {
                // block_q3_K: hmask[32] | qs[64] | scales[12] | d (f16).
                // Value p's code is its low bits `qs[32 (p / 128) + p % 32]
                // >> 2 (p % 128 / 32)` and its high bit `hmask[p % 32] >>
                // (p / 32)`, 3 bits a value; ggml's `q - 4 (1 - h)` is that
                // code offset by 4.
                push(plane(
                    "words",
                    PlaneRepackRecipe::BitRoutes(
                        (0..256 * 3)
                            .map(|bit| {
                                let (position, code_bit) = (bit / 3, bit % 3);
                                if code_bit < 2 {
                                    (32 + (position / 128) * 32 + position % 32) * 8
                                        + (position % 128 / 32) * 2
                                        + code_bit
                                } else {
                                    (position % 32) * 8 + position / 32
                                }
                            })
                            .collect(),
                    ),
                ));
                // Sub-block j's six-bit scale code (offset by 32): low four
                // bits `scales[j % 8] >> 4 (j / 8)`, high two bits
                // `scales[8 + j % 4] >> 2 (j / 4)`.
                push(plane(
                    "coefficients",
                    PlaneRepackRecipe::BitRoutes(
                        (0..16 * 6)
                            .map(|bit| {
                                let (j, code_bit) = (bit / 6, bit % 6);
                                if code_bit < 4 {
                                    (96 + j % 8) * 8 + 4 * (j / 8) + code_bit
                                } else {
                                    (104 + j % 4) * 8 + 2 * (j / 4) + code_bit - 4
                                }
                            })
                            .collect(),
                    ),
                ));
                push(plane(
                    "scale_factor",
                    PlaneRepackRecipe::BitRoutes(direct_bits(108, 2)),
                ));
            }
            "gguf_iq3_s" => {
                // block_iq3_s: d (f16) | qs[64] | qh[8] | signs[32] |
                // scales[4]. Entry `e` (values 8e..8e+8, its two four-value
                // grid points) is its 9-bit indices `qs[2e + i] | (qh[e / 4]
                // >> 2 (e % 4) + i & 1) << 8`, first lowest, then its signs
                // `signs[e]`, bit for bit.
                push(plane(
                    "words",
                    PlaneRepackRecipe::BitRoutes(
                        (0..32 * 26)
                            .map(|bit| {
                                let (entry, bit) = (bit / 26, bit % 26);
                                match bit {
                                    0..18 => {
                                        let (index, bit) = (bit / 9, bit % 9);
                                        if bit < 8 {
                                            (2 + 2 * entry + index) * 8 + bit
                                        } else {
                                            (66 + entry / 4) * 8 + 2 * (entry % 4) + index
                                        }
                                    }
                                    _ => (74 + entry) * 8 + bit - 18,
                                }
                            })
                            .collect(),
                    ),
                ));
                // Sub-block b's 4-bit scale code `scales[b / 2] >> 4 (b % 2)`
                // (ggml's `1 + 2 s`, the odd code).
                push(plane(
                    "coefficients",
                    PlaneRepackRecipe::BitRoutes(direct_bits(106, 4)),
                ));
                push(plane(
                    "scale_factor",
                    PlaneRepackRecipe::BitRoutes(direct_bits(0, 2)),
                ));
            }
            "gguf_q4_0" | "gguf_iq4_nl" | "gguf_q4_1" => {
                // block_q4_0 and block_iq4_nl: d (f16) | qs[16]; block_q4_1:
                // d | m (f16) | qs[16]. Value p is the low (p < 16) or high
                // nibble of qs[p % 16]; d and m move bit for bit.
                let minimum = source_name == "gguf_q4_1";
                let header = if minimum { 4 } else { 2 };
                push(plane(
                    "words",
                    PlaneRepackRecipe::BitRoutes(
                        (0..128)
                            .map(|bit| {
                                let position = bit / 4;
                                (header + position % 16) * 8 + (position / 16) * 4 + bit % 4
                            })
                            .collect(),
                    ),
                ));
                push(plane(
                    "scale",
                    PlaneRepackRecipe::BitRoutes(direct_bits(0, 2)),
                ));
                if minimum {
                    push(plane(
                        "bias",
                        PlaneRepackRecipe::BitRoutes(direct_bits(2, 2)),
                    ));
                }
            }
            "gguf_mxfp4" => {
                // block_mxfp4: e (E8M0) | qs[16]. Value p is the low (p < 16)
                // or high nibble of qs[p % 16], with ggml's +0 for code 8;
                // e moves bit for bit.
                push(plane(
                    "words",
                    PlaneRepackRecipe::PackedEntries(
                        (0..32)
                            .map(|p| e2m1_positive_zero((1 + p % 16) * 8 + (p / 16) * 4))
                            .collect(),
                    ),
                ));
                push(plane(
                    "block_scale",
                    PlaneRepackRecipe::BitRoutes(direct_bits(0, 1)),
                ));
            }
            "gguf_q5_0" | "gguf_q5_1" => {
                // block_q5_0: d (f16) | qh[4] | qs[16]; block_q5_1: d | m
                // (f16) | qh[4] | qs[16]. Value p: the low (p < 16) or high
                // nibble of qs[p % 16], and bit p of qh as its fifth bit.
                let minimum = source_name == "gguf_q5_1";
                let high = if minimum { 4 } else { 2 };
                push(plane(
                    "words",
                    PlaneRepackRecipe::BitRoutes(
                        (0..160)
                            .map(|bit| {
                                let (position, code_bit) = (bit / 5, bit % 5);
                                if code_bit < 4 {
                                    (high + 4 + position % 16) * 8 + (position / 16) * 4 + code_bit
                                } else {
                                    high * 8 + position
                                }
                            })
                            .collect(),
                    ),
                ));
                push(plane(
                    "scale",
                    PlaneRepackRecipe::BitRoutes(direct_bits(0, 2)),
                ));
                if minimum {
                    push(plane(
                        "bias",
                        PlaneRepackRecipe::BitRoutes(direct_bits(2, 2)),
                    ));
                }
            }
            "gguf_nvfp4" => {
                // block_nvfp4: d[4] (UE4M3) | qs[32]. Value p of sixteen-value
                // sub-block s = p / 16, w = p % 16, is the low (w < 8) or high
                // nibble of qs[8 s + w % 8], with ggml's +0 for code 8; scale
                // s is d[s].
                push(plane(
                    "words",
                    PlaneRepackRecipe::PackedEntries(
                        (0..64)
                            .map(|p| {
                                let (block, within) = (p / 16, p % 16);
                                e2m1_positive_zero(
                                    (4 + 8 * block + within % 8) * 8 + (within / 8) * 4,
                                )
                            })
                            .collect(),
                    ),
                ));
                push(plane(
                    "block_scale",
                    PlaneRepackRecipe::BitRoutes(direct_bits(0, 4)),
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
        source: &ExternalPacketLayout,
        destination: &PackedPacketLayout,
        recipe: &PacketRepackRecipe,
    ) {
        assert!(source.packet_size > 0, "external packet size is zero");
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
                RepackExpr::F16ToF32(value) => {
                    assert!(
                        matches!(**value, RepackExpr::SourceBits { width: 16, .. }),
                        "representation conversion widens something other than a binary16 source field"
                    );
                    validate_expr(value, source_bits);
                }
                RepackExpr::ShiftLeft { value, .. } => validate_expr(value, source_bits),
                RepackExpr::ScaleF32 { value, exponent } => {
                    // Every widened binary16 times 2^±8 is a normal f32.
                    assert!(
                        matches!(**value, RepackExpr::F16ToF32(_)) && exponent.unsigned_abs() <= 8,
                        "representation conversion scales something other than a widened binary16 field exactly"
                    );
                    validate_expr(value, source_bits);
                }
                RepackExpr::Lookup { index, table } => {
                    validate_expr(index, source_bits);
                    let (min, max) = integer_range(index);
                    assert!(
                        min >= 0 && max < table.values().len() as i64,
                        "representation conversion indexes outside its lookup table"
                    );
                }
                RepackExpr::BitOr(left, right) => {
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
                PlaneRepackRecipe::PackedEntries(values) => {
                    assert_eq!(
                        u32::try_from(values.len())
                            .ok()
                            .and_then(|entries| entries.checked_mul(plane.entry_bits)),
                        Some(plane_bits),
                        "packed-entry conversion does not initialize its entire plane"
                    );
                    // Integer codes, or the raw payloads of a floating code.
                    let (bits, low, high) = match &plane.encoding {
                        PlaneEncoding::Packed {
                            bits,
                            interpretation: CodeInterpretation::TwosComplement,
                        } => (*bits, -(1i64 << (bits - 1)), (1i64 << (bits - 1)) - 1),
                        PlaneEncoding::Packed { bits, .. } => (*bits, 0, (1i64 << bits) - 1),
                        PlaneEncoding::FloatCode { format } => {
                            (format.bits(), 0, (1i64 << format.bits()) - 1)
                        }
                        PlaneEncoding::Dense(_) => {
                            panic!("packed-entry conversion recipe targets a dense plane")
                        }
                    };
                    for expression in values {
                        validate_expr(expression, source_bits);
                        let (min, max) = integer_range(expression);
                        assert!(
                            low <= min && max <= high,
                            "packed-entry conversion value range [{min}, {max}] exceeds its {bits}-bit entry"
                        );
                    }
                }
            }
        }
    }

    /// A bound on the values of an integer repack expression.
    fn integer_range(expression: &RepackExpr) -> (i64, i64) {
        match expression {
            RepackExpr::SourceBits { width, .. } => (0, (1i64 << width) - 1),
            RepackExpr::ShiftLeft { value, bits } => {
                let (min, max) = integer_range(value);
                (min << bits, max << bits)
            }
            RepackExpr::BitOr(left, right) => {
                let (left_min, left_max) = integer_range(left);
                let (right_min, right_max) = integer_range(right);
                assert!(
                    left_min >= 0 && right_min >= 0,
                    "a packed-entry OR combines non-negative bit fields"
                );
                (left_min.max(right_min), left_max + right_max)
            }
            RepackExpr::Lookup { table, .. } => (
                i64::from(
                    *table
                        .values()
                        .iter()
                        .min()
                        .expect("a lookup table has entries"),
                ),
                i64::from(
                    *table
                        .values()
                        .iter()
                        .max()
                        .expect("a lookup table has entries"),
                ),
            ),
            RepackExpr::F16ToF32(_) | RepackExpr::ScaleF32 { .. } => {
                panic!("a packed-entry conversion value is floating")
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
    pub(super) fn intrinsic_overloads(
        capability: CapabilityId,
        name: &str,
    ) -> &'static [IntrinsicId] {
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
        representation_conversions()
            .iter()
            .find(|conversion| conversion.source == source && conversion.destination == destination)
    }
    pub(super) fn representation_conversions() -> &'static [RepresentationConversion] {
        let tables = tables();
        tables
            .conversions
            .get_or_init(|| conversions(&tables.representations))
    }
    pub(super) fn representation_conversion_info(
        id: RepresentationConversionId,
    ) -> &'static RepresentationConversion {
        representation_conversions().get(id.index()).unwrap_or_else(|| {
            panic!("StaticRegistry produced a RepresentationConversionId outside its conversion table (§13.3.1)")
        })
    }
    pub(super) fn dense(dtype: DType) -> RepresentationId {
        RepresentationId::new(u32::from(dtype.ordinal()))
    }
    pub(super) fn table_len(table: Table) -> usize {
        let t = tables();
        match table {
            Table::Capability => t.capabilities.len(),
            Table::Intrinsic => t.intrinsics.len(),
            Table::Representation => t.representations.len(),
            Table::RepresentationConversion => representation_conversions().len(),
        }
    }
    pub(super) fn declared_name(name: &str) -> Option<&'static str> {
        static NAMES: OnceLock<std::collections::HashSet<&'static str>> = OnceLock::new();
        NAMES
            .get_or_init(|| {
                let mut names = std::collections::HashSet::new();
                for signature in &tables().intrinsics {
                    names.insert(signature.name);
                    if let IntrinsicResultType::Opaque { name, .. } = signature.result {
                        names.insert(name);
                    }
                    for argument in &signature.arguments {
                        if let OperandCategory::Opaque { name, .. } = argument.category {
                            names.insert(name);
                        }
                    }
                }
                names
            })
            .get(name)
            .copied()
    }

    pub(super) fn intrinsic_denotation(id: IntrinsicId) -> IntrinsicDenotation {
        *tables().denotations.get(id.index()).unwrap_or_else(|| {
            panic!("StaticRegistry produced an IntrinsicId outside its denotation table (§13.3.1)")
        })
    }

    // ----- crate-private views -------------------------------------------

    /// The packed representation behind a packed storage id (any layout).
    pub(crate) fn packed(id: RepresentationId) -> Option<&'static Repr> {
        let info = representation_info(id);
        match info.kind {
            RepresentationKind::Packed(_) | RepresentationKind::PackedRows(_) => {
                crate::repr::lookup(info.representation)
            }
            RepresentationKind::Dense(_) | RepresentationKind::External(_) => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Code planes split a width into a power-of-two low plane and a high
    /// plane of at most 2 bits; 7 and 11 bits have no split.
    #[test]
    fn code_planes_split_every_width() {
        let planes: Vec<_> = (1..=11).map(super::code_planes).collect();
        assert_eq!(
            planes,
            [
                Some((1, 0)),
                Some((2, 0)),
                Some((2, 1)),
                Some((4, 0)),
                Some((4, 1)),
                Some((4, 2)),
                None,
                Some((8, 0)),
                Some((8, 1)),
                Some((8, 2)),
                None,
            ]
        );
    }

    #[test]
    fn tables_intern_every_row_once() {
        let rows = crate::intrinsics::capability_rows();
        let mut interned = 0;
        for backend in BackendName::ALL {
            for info in capabilities(backend) {
                assert_eq!(info.backend, backend);
                assert_eq!(capability(backend, info.name), Some(info.id));
                for signature in intrinsics(info.id) {
                    assert_eq!(signature.capability, info.id);
                    interned += 1;
                }
            }
        }
        assert_eq!(interned, rows.len());
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
        assert_eq!(
            intrinsic_denotation(lane),
            IntrinsicDenotation::ParticipantIndex
        );
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
                    assert_eq!(
                        plane_element_dtype(info.id, plane.name),
                        plane.storage_dtype
                    );
                }
            }
        }
    }

    #[test]
    fn narrow_float_encodings_cover_subnormals_and_finite_roundtrips() {
        assert_eq!(f16_to_f32(1), 2.0f32.powi(-24));
        assert_eq!(f16_to_f32(0x03ff), 1023.0 * 2.0f32.powi(-24));
        for bits in 0..=u16::MAX {
            let sign = u32::from(bits & 0x8000) << 16;
            let exponent = i32::from((bits >> 10) & 0x1f);
            let mantissa = u32::from(bits & 0x03ff);
            let exact = match exponent {
                0x1f => 0x7f80_0000 | mantissa << 13 | sign,
                0 => (mantissa as f32 * 2f32.powi(-24)).to_bits() | sign,
                _ => ((1024 + mantissa) as f32 * 2f32.powi(exponent - 25)).to_bits() | sign,
            };
            let value = f16_to_f32(bits);
            assert_eq!(value.to_bits(), exact, "half widening {bits:04x}");
            if value.is_nan() {
                let encoded = f16_bits(value);
                assert!(f16_to_f32(encoded).is_nan(), "half NaN {bits:04x}");
                assert_eq!(encoded & 0x8000, bits & 0x8000, "half NaN sign {bits:04x}");
            } else {
                assert_eq!(f16_bits(value), bits, "half encoding {bits:04x}");
            }
        }
    }

    #[test]
    fn half_encoding_rounds_to_nearest_even() {
        let cases: [(f32, u16); 14] = [
            // Ties between 1 and its successor go to the even payload.
            (1.0 + 2f32.powi(-11), 0x3c00),
            (1.0 + 3.0 * 2f32.powi(-11), 0x3c02),
            (1.0 + 2f32.powi(-11) + 2f32.powi(-20), 0x3c01),
            // A mantissa carry adds into the exponent, including from an odd one.
            (2.0 - 2f32.powi(-12), 0x4000),
            (-(2.0 - 2f32.powi(-12)), 0xc000),
            (4.0 - 2f32.powi(-11), 0x4400),
            // Largest finite, the overflow tie, and past it.
            (65_504.0, 0x7bff),
            (65_519.0, 0x7bff),
            (65_520.0, 0x7c00),
            // Subnormal ties and the boundary into the normal range.
            (2f32.powi(-25), 0x0000),
            (1.5 * 2f32.powi(-25), 0x0001),
            (3.0 * 2f32.powi(-25), 0x0002),
            (2f32.powi(-14) - 2f32.powi(-25), 0x0400),
            (-2f32.powi(-26), 0x8000),
        ];
        for (value, bits) in cases {
            assert_eq!(f16_bits(value), bits, "{value:e}");
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

    /// A conversion's domain makes the scales it converts as numbers finite
    /// and keeps every other source bit: IQ2_XS's factor, and nothing of
    /// Q4_0 or IQ4_XS, whose scales move bit for bit.
    #[test]
    fn a_conversion_admits_sources_with_finite_converted_scales() {
        for (source, destination, fields) in [
            ("gguf_iq2_xs", "iq2xs@rows16", vec![0u32]),
            ("gguf_iq4_xs", "iq4xs@rows16", vec![]),
            ("gguf_q4_0", "q4g32s@rows16", vec![]),
        ] {
            let conversion = representation_conversion(
                representation(source).unwrap(),
                representation(destination).unwrap(),
            )
            .unwrap();
            assert_eq!(conversion.recipe.converted_f16_fields(), fields, "{source}");
            let RepresentationKind::External(layout) = &representation_info(conversion.source).kind
            else {
                panic!("{source} is external storage")
            };
            let mut packets = vec![0xffu8; 2 * layout.packet_size as usize];
            conversion.admit_source(&mut packets);
            for (packet, bytes) in packets
                .chunks_exact(layout.packet_size as usize)
                .enumerate()
            {
                for (index, byte) in bytes.iter().enumerate() {
                    let scale_high = fields.iter().any(|field| index as u32 == field / 8 + 1);
                    assert_eq!(
                        *byte,
                        if scale_high { 0xbf } else { 0xff },
                        "{source} packet {packet} byte {index}"
                    );
                }
            }
        }
    }

    #[test]
    fn resident_conversions_are_keyed_by_source_and_destination() {
        let q4_k = representation("gguf_q4_k").unwrap();
        // The destination's packet form, or its row layout's tile rows.
        for (layout, name, tile_rows) in [
            (Layout::Packet, "q4k", None),
            (Layout::Rows16, "q4k@rows16", Some(1)),
            (Layout::Mma16, "q4k@mma16", Some(16)),
        ] {
            let destination = storage("q4k", layout).unwrap();
            assert_eq!(destination, representation(name).unwrap());
            let conversion = representation_conversion(q4_k, destination).unwrap();
            match conversion.destination_layout {
                ConversionLayout::Packet(packet) => {
                    assert_eq!(tile_rows, None);
                    assert_eq!(
                        RepresentationKind::Packed(packet.clone()),
                        representation_info(destination).kind
                    );
                }
                ConversionLayout::Rows(rows) => {
                    assert_eq!(Some(rows.tile_rows()), tile_rows);
                    assert_eq!(rows.layout, layout);
                }
            }
        }
        // An external source converts into every element of its resident
        // representation, each recipe in every layout, once; nothing else
        // converts.
        for info in representations() {
            let destinations = representations()
                .iter()
                .filter(|destination| representation_conversion(info.id, destination.id).is_some())
                .collect::<Vec<_>>();
            let RepresentationKind::External(_) = info.kind else {
                assert!(destinations.is_empty(), "`{}`", info.name);
                continue;
            };
            let resident = destinations[0].representation;
            let elements = representations()
                .iter()
                .filter(|element| element.representation == resident)
                .map(|element| element.id)
                .collect::<Vec<_>>();
            let destinations = destinations
                .iter()
                .map(|element| element.id)
                .collect::<Vec<_>>();
            assert_eq!(destinations, elements, "`{}`", info.name);
        }
        let nvfp4 = representation("gguf_nvfp4").unwrap();
        for layout in Layout::ALL {
            for (storage_name, operands) in [
                ("nvfp4g16", OperandRecipe::Rounded),
                ("nvfp4g16a4", OperandRecipe::Nvfp4Rows),
            ] {
                let element = storage(storage_name, layout).unwrap();
                assert_eq!(representation_info(element).operands, operands);
                assert!(representation_conversion(nvfp4, element).is_some());
            }
        }
        assert_eq!(storage("f32", Layout::Rows16), None);
        assert_eq!(storage("q4g64", Layout::Rows16), None);
    }

    fn row_layout(name: &str) -> &'static PackedRowLayout {
        let RepresentationKind::PackedRows(layout) =
            &representation_info(representation(name).unwrap()).kind
        else {
            panic!("`{name}` is a row layout")
        };
        layout
    }

    #[test]
    fn rows16_planes_follow_the_frozen_interface() {
        let plane_set = |name: &str| {
            row_layout(name)
                .planes
                .iter()
                .map(|plane| (plane.name, plane.bytes_per_group))
                .collect::<Vec<_>>()
        };
        assert_eq!(
            plane_set("q4k@rows16"),
            [("codes_lo", 128), ("scales", 12), ("supers", 4)]
        );
        assert_eq!(
            plane_set("q5k@rows16"),
            [
                ("codes_lo", 128),
                ("codes_hi", 32),
                ("scales", 12),
                ("supers", 4)
            ]
        );
        assert_eq!(
            plane_set("q6k@rows16"),
            [
                ("codes_lo", 128),
                ("codes_hi", 64),
                ("scales", 16),
                ("supers", 2)
            ]
        );
        assert_eq!(plane_set("q8g32s@rows16"), [("codes", 32), ("supers", 2)]);
        assert_eq!(
            plane_set("iq4xs@rows16"),
            [("codes_lo", 128), ("scales", 6), ("supers", 2)]
        );
        // Qwen3.5 K = 2560: codes 1280 | scales 120 -> 128 | supers 40 -> 48.
        let q4k = row_layout("q4k@rows16");
        assert_eq!(q4k.plane_row_offset(1, 2560), Some(1280));
        assert_eq!(q4k.plane_row_offset(2, 2560), Some(1408));
        assert_eq!(q4k.row_stride_bytes(2560), Some(1456));
        assert_eq!(
            canonical_bytes(representation("q4k@rows16").unwrap(), &[3, 2560]),
            Some(3 * 1456)
        );
        // A q8 row of three groups: 96 code bytes, 6 super bytes -> 16.
        let q8 = row_layout("q8g32s@rows16");
        assert_eq!(q8.row_stride_bytes(96), Some(112));
        // mma16 pads rows per matrix and q8 rows to whole 64-column k-blocks.
        let q8_mma = representation("q8g32s@mma16").unwrap();
        assert_eq!(
            row_layout("q8g32s@mma16").row_stride_bytes(96),
            Some(128 + 16)
        );
        assert_eq!(canonical_bytes(q8_mma, &[2, 17, 96]), Some(2 * 32 * 144));
        assert_eq!(canonical_bytes(q8_mma, &[96]), None);
        // mma16 rows hold whole k-blocks and whole 16-byte chunks of every
        // code plane: a 32-value group's one-bit high plane (4 bytes) takes
        // groups in fours, so Gemma's K = 704 stores 24 groups.
        for (name, multiple) in [
            ("q4k@mma16", 1),
            ("q6k@mma16", 1),
            ("iq4xs@mma16", 1),
            ("q8g32s@mma16", 2),
            ("q4g32s@mma16", 2),
            ("mxfp4g32@mma16", 2),
            ("nvfp4g16@mma16", 1),
            ("q5g32s@mma16", 4),
            ("q5g32@mma16", 4),
            ("q4g32@mma16", 2),
        ] {
            assert_eq!(row_layout(name).group_multiple, multiple, "{name}");
        }
        assert_eq!(row_layout("q5g32@mma16").row_groups(704), Some(24));
        assert_eq!(row_layout("q5g32@rows16").row_groups(704), Some(22));
    }

    /// `mma16` code placement re-derived from the PTX m16n8k16 A fragment:
    /// lane `4g + t` register `i` holds (row `g + 8 (i & 1)`, k
    /// `2t + 8 (i >> 1) + {0, 1}`), and nibbles `i` and `i + 4` of a lane's
    /// step word are that register's pair.
    #[test]
    fn mma16_code_placement_is_the_a_fragment_order() {
        let layout = row_layout("q4k@mma16");
        let k = 256u64;
        let geometry = layout.geometry(k).unwrap();
        let stride = geometry.stride;
        let row_bytes = geometry.bytes_per_row[0];
        let mut seen = std::collections::HashSet::new();
        for row in 0..16u64 {
            for column in 0..k {
                let bit = layout.code_bit(&geometry, 0, row, column);
                // Every code lands inside some row's codes_lo payload.
                assert!(bit / 8 % stride < row_bytes);
                assert!(seen.insert(bit), "codes collide");
            }
        }
        for lane in 0..32u64 {
            let (g, t) = (lane / 4, lane % 4);
            for block in 0..k / 64 {
                for step in 0..4u64 {
                    for register in 0..4u64 {
                        for half in 0..2u64 {
                            let row = g + 8 * (register & 1);
                            let column =
                                64 * block + 16 * step + 2 * t + 8 * (register >> 1) + half;
                            let v =
                                (block * 32 + lane) * 128 + 32 * step + 4 * (register + 4 * half);
                            let expected =
                                (v / 8 / row_bytes * stride + v / 8 % row_bytes) * 8 + v % 8;
                            assert_eq!(layout.code_bit(&geometry, 0, row, column), expected);
                        }
                    }
                }
            }
        }
    }

    /// A `grids` plane keeps each entry's bytes together and interleaves
    /// whole chunks of a row's columns across a tile: in `mma16` a tile's
    /// k-block is its 16 rows' 64 columns, row by row, inside the rows' own
    /// plane payloads; in `rows32` a chunk is its 32 rows' 32 columns.
    #[test]
    fn grids_interleave_whole_entries_row_by_row() {
        for (name, entry_bytes, layouts) in [
            ("iq2xs", 2u64, &["mma16", "rows32"][..]),
            ("iq3xxs", 3, &["mma16", "rows32"]),
            ("iq2xxs", 2, &["rows32"]),
        ] {
            for (layout_name, tile, chunk) in [("mma16", 16u64, 64u64), ("rows32", 32, 32)]
                .into_iter()
                .filter(|(layout_name, ..)| layouts.contains(layout_name))
            {
                let layout = row_layout(&format!("{name}@{layout_name}"));
                let k = 512u64;
                let geometry = layout.geometry(k).unwrap();
                let (stride, offset) = (geometry.stride, geometry.offsets[0]);
                let row_bytes = geometry.bytes_per_row[0];
                for row in 0..2 * tile {
                    for column in (0..k).step_by(8) {
                        let unit = chunk / 8 * entry_bytes;
                        let v = (column / chunk * tile + row % tile) * unit
                            + column % chunk / 8 * entry_bytes;
                        let expected = if layout_name == "mma16" {
                            (row / tile * tile + v / row_bytes) * stride + offset + v % row_bytes
                        } else {
                            row / tile * tile * stride + offset * tile + v
                        };
                        assert_eq!(layout.code_bit(&geometry, 0, row, column), expected * 8);
                    }
                }
            }
        }
    }

    /// A carried grid representation's entries hold its coefficient codes
    /// (bit `j` of a group's code at bit 15 of the group's entry `j`); its
    /// row layouts have no `scales` plane, and its `mma16` tile is the
    /// superblock records of its 16 rows over the tile's rows whole: every
    /// group's entries in GEMV lane order, 1024 bytes apart, then every
    /// group's rows' f16 factors.
    #[test]
    fn carried_grids_hold_their_scales_and_mma16_records_their_factors() {
        let plane_set = |name: &str| {
            row_layout(name)
                .planes
                .iter()
                .map(|plane| (plane.name, plane.bytes_per_group))
                .collect::<Vec<_>>()
        };
        for layout in ["rows16", "rows8", "rows32"] {
            assert_eq!(
                plane_set(&format!("iq2xxs@{layout}")),
                [("grids", 64), ("supers", 2)]
            );
        }
        assert_eq!(plane_set("iq2xxs@mma16"), [("records", 66)]);
        let layout = row_layout("iq2xxs@mma16");
        // Bolt's shared projections: K = 4096 holds 16 records a tile and
        // rows of 1056 bytes; K = 5376 rows of 21 * 66 = 1386 -> 1392.
        assert_eq!(layout.row_stride_bytes(4096), Some(1056));
        assert_eq!(layout.row_stride_bytes(5376), Some(1392));
        let k = 512u64;
        let geometry = layout.geometry(k).unwrap();
        let stride = geometry.stride;
        let (entries, groups) = (16 * 64, k / 256);
        for row in 0..32u64 {
            let (tile, r) = (row / 16, row % 16);
            let (g, half) = (r % 8, r / 8);
            for column in (0..k).step_by(8) {
                let (p, q, e) = (column / 256, column % 256 / 64, column % 64 / 8);
                let lane = 4 * g + e % 4;
                let v = p * entries + q / 2 * 512 + lane * 16 + (q % 2 * 2 + half) * 4 + e / 4 * 2;
                assert_eq!(
                    layout.code_bit(&geometry, 0, row, column),
                    (tile * 16 * stride + v) * 8
                );
            }
            for p in 0..k / 256 {
                // Factor bit b of row r: word g of group p's factors, past
                // every group's entries.
                for bit in 0..16u32 {
                    let v = groups * entries + p * 32 + 4 * g + 2 * half;
                    assert_eq!(
                        layout.packet_bit(&geometry, row, p, 2, bit),
                        (tile * 16 * stride + v) * 8 + u64::from(bit)
                    );
                }
                // Scale code bit j of group f: bit 15 of entry 4 f + j.
                for bit in 0..32u32 {
                    let (f, j) = (u64::from(bit / 4), u64::from(bit % 4));
                    let column = p * 256 + (4 * f + j) * 8;
                    assert_eq!(
                        layout.packet_bit(&geometry, row, p, 1, bit),
                        layout.code_bit(&geometry, 0, row, column) + 15
                    );
                }
            }
        }
        let rows = row_layout("iq2xxs@rows16");
        let geometry = rows.geometry(k).unwrap();
        for bit in 0..32u32 {
            let (f, j) = (u64::from(bit / 4), u64::from(bit % 4));
            assert_eq!(
                rows.packet_bit(&geometry, 3, 1, 1, bit),
                rows.code_bit(&geometry, 0, 3, 256 + (4 * f + j) * 8) + 15
            );
        }
    }

    /// Deterministic canonical bytes of an external tensor.
    fn source_bytes(source: RepresentationId, shape: &[u64], seed: u32) -> Vec<u8> {
        let length = canonical_bytes(source, shape).unwrap() as usize;
        (0..length)
            .map(|index| {
                let x = (index as u32)
                    .wrapping_add(seed)
                    .wrapping_mul(2_654_435_761);
                (x >> 13) as u8
            })
            .collect()
    }

    /// Bit-exact round trip for every (format, layout): decoding the
    /// converted storage in any layout equals decoding the packet form
    /// converted from the same source, including partial packets, a row
    /// count off the 16-row tile, and matrices of a rank-3 tensor.
    #[test]
    fn every_layout_decodes_exactly_like_the_packet_form() {
        use crate::interp::{TensorData, repack};
        for source_info in representations()
            .iter()
            .filter(|info| matches!(info.kind, RepresentationKind::External(_)))
        {
            let RepresentationKind::External(external) = &source_info.kind else {
                unreachable!()
            };
            let group = u64::from(external.logical_group);
            for shape in [
                vec![17, 3 * group],
                vec![2, 3, 2 * group - 8],
                vec![1, group],
            ] {
                let host_shape = shape
                    .iter()
                    .map(|extent| *extent as usize)
                    .collect::<Vec<_>>();
                let bytes = source_bytes(source_info.id, &shape, shape[0] as u32);
                let decode = |conversion: &RepresentationConversion| {
                    let converted = repack(conversion.id, &host_shape, &bytes).unwrap();
                    let data = TensorData::encoded(
                        conversion.destination,
                        host_shape.clone(),
                        converted.clone(),
                    )
                    .unwrap();
                    (converted, data.values().unwrap())
                };
                for packet in representations()
                    .iter()
                    .filter(|destination| destination.layout == Layout::Packet)
                    .filter_map(|destination| representation_conversion(source_info.id, destination.id))
                {
                    let (packet_bytes, expected) = decode(packet);
                    let element = representation_info(packet.destination).name;
                    for layout in [Layout::Rows16, Layout::Rows8, Layout::Mma16, Layout::Rows32] {
                        let conversion = representation_conversion(
                            source_info.id,
                            storage(element, layout).unwrap(),
                        )
                        .unwrap();
                        let (converted, actual) = decode(conversion);
                        let RepresentationKind::PackedRows(rows) =
                            &representation_info(conversion.destination).kind
                        else {
                            unreachable!()
                        };
                        assert_eq!(rows.packets(&shape, &converted), packet_bytes);
                        assert_eq!(rows.place(&shape, &packet_bytes), converted);
                        assert_eq!(
                            actual
                                .iter()
                                .map(|value| value.to_bits())
                                .collect::<Vec<_>>(),
                            expected
                                .iter()
                                .map(|value| value.to_bits())
                                .collect::<Vec<_>>(),
                            "{} -> {} over {shape:?}",
                            source_info.name,
                            representation_info(conversion.destination).name
                        );
                    }
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
            assert!(
                canonical
                    .iter()
                    .zip(&occupied)
                    .all(|(byte, occupied)| *occupied || *byte == 0)
            );
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
            assemble(&[
                ("words", &words),
                ("scale", &scale),
                ("bias", &bias),
                ("extra", &bias)
            ]),
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
