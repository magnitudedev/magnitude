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

/// Revision of the whole registry. Any semantic change to a primitive,
/// capability, intrinsic, or representation changes this string, and with it
/// every cache identity.
pub const REGISTRY_REVISION: &str = "seismic-registry-v15";

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
/// - `Mma16`: `Rows16` whose code-plane bytes are permuted, per 16-row tile,
///   into `mma.sync.m16n8k16` A-fragment order (`PackedRowLayout::code_bit`).
///   The CUDA resident layout.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Layout {
    Packet,
    Rows16,
    Mma16,
}

impl Layout {
    pub const ALL: [Layout; 3] = [Layout::Packet, Layout::Rows16, Layout::Mma16];

    pub const fn as_str(self) -> &'static str {
        match self {
            Layout::Packet => "packet",
            Layout::Rows16 => "rows16",
            Layout::Mma16 => "mma16",
        }
    }

    pub fn parse(name: &str) -> Option<Layout> {
        Layout::ALL.into_iter().find(|layout| layout.as_str() == name)
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

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ExternalPacketLayout {
    pub packet_size: u32,
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
    /// Source packet -> the destination representation's packet form.
    pub recipe: PacketRepackRecipe,
    /// How converted packets are placed in the destination layout.
    pub kind: ConversionKind,
}

/// The placement of a conversion's packets in its destination layout.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ConversionKind {
    /// One source packet becomes one resident packet (`packet` layout).
    Packet,
    /// One source packet becomes one group column of each plane of its row
    /// (`rows16`).
    Row,
    /// The source packets of `rows` rows at one group column become one
    /// row tile, whose code planes are permuted across those rows (`mma16`).
    RowTile { rows: u32 },
}

/// Exact, validated ownership for one external packet -> resident packet-form
/// conversion. There is exactly one recipe for every packet-form plane and
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
    /// `Rows16` or `Mma16`.
    pub layout: Layout,
    /// The representation's packet form: logical group, plane schema and the
    /// packet every conversion produces before placement.
    pub packet: PackedPacketLayout,
    /// Row planes in storage order.
    pub planes: Vec<RowPlaneInfo>,
    /// Storage groups per row round up to a multiple of this (`mma16`
    /// stores whole 64-column k-blocks).
    pub group_multiple: u32,
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
    /// The bytes of the listed packet planes of one packet, concatenated in
    /// list order, per storage group.
    Groups { planes: Vec<u32> },
}

impl PackedRowLayout {
    /// Logical values per storage group.
    pub fn group(&self) -> u32 {
        self.packet.group
    }

    /// Rows of one `mma16` tile, or 1.
    pub fn tile_rows(&self) -> u64 {
        match self.layout {
            Layout::Mma16 => MMA_TILE_ROWS,
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
        if self.layout == Layout::Mma16 {
            let rows = units.last_mut()?;
            *rows = rows.div_ceil(MMA_TILE_ROWS).checked_mul(MMA_TILE_ROWS)?;
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
            Layout::Mma16 => {
                let rows = extents[extents.len() - 2];
                row / rows * rows.div_ceil(MMA_TILE_ROWS) * MMA_TILE_ROWS + row % rows
            }
            Layout::Rows16 => row,
            Layout::Packet => unreachable!("a row layout is never `packet`"),
        }
    }

    /// Absolute bit of the first bit of the code of column `column` of stored
    /// row `row` in code plane `plane`, for rows of `logical_extent` values.
    ///
    /// `rows16`: bit `column * bits` of the row's plane.
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
    /// is the f16x2 code pair of A register `i`.
    pub fn code_bit(&self, geometry: &RowGeometry, plane: usize, row: u64, column: u64) -> u64 {
        let RowPlaneContent::Codes { bits, .. } = self.planes[plane].content else {
            panic!("row plane `{}` holds no codes", self.planes[plane].name)
        };
        let bits = u64::from(bits);
        let stride = geometry.stride;
        let offset = geometry.offsets[plane];
        match self.layout {
            Layout::Rows16 => (row * stride + offset) * 8 + column * bits,
            Layout::Mma16 => {
                let row_bytes = geometry.bytes_per_row[plane];
                let (tile, r) = (row / MMA_TILE_ROWS, row % MMA_TILE_ROWS);
                let (block, c) = (column / MMA_KBLOCK, column % MMA_KBLOCK);
                let (step, k) = (c / 16, c % 16);
                let lane = 4 * (r % 8) + (k % 8) / 2;
                let slot = (k / 8) * 2 + r / 8 + 4 * (k % 2);
                let bit = (block * MMA_LANES + lane) * 32 * bits + 8 * bits * step + bits * slot;
                let (byte, within) = (bit / 8, bit % 8);
                let stored = tile * MMA_TILE_ROWS + byte / row_bytes;
                (stored * stride + offset + byte % row_bytes) * 8 + within
            }
            Layout::Packet => unreachable!("a row layout is never `packet`"),
        }
    }

    /// Absolute bit holding bit `bit` of packet-form plane `plane` of packet
    /// (storage group) `packet` of stored row `row`.
    pub fn packet_bit(
        &self,
        geometry: &RowGeometry,
        row: u64,
        packet: u64,
        plane: usize,
        bit: u32,
    ) -> u64 {
        if plane == 0 {
            let code_bits = self.packet.planes[0].entry_bits;
            let (code, code_bit) = (bit / code_bits, bit % code_bits);
            let (index, shift) = self
                .planes
                .iter()
                .enumerate()
                .find_map(|(index, candidate)| match candidate.content {
                    RowPlaneContent::Codes { shift, bits }
                        if (shift..shift + bits).contains(&code_bit) =>
                    {
                        Some((index, shift))
                    }
                    _ => None,
                })
                .expect("every code bit has one row code plane");
            let column = packet * u64::from(self.packet.group) + u64::from(code);
            return self.code_bit(geometry, index, row, column) + u64::from(code_bit - shift);
        }
        let (index, prefix) = self
            .planes
            .iter()
            .enumerate()
            .find_map(|(index, candidate)| match &candidate.content {
                RowPlaneContent::Groups { planes } => {
                    let position = planes.iter().position(|member| *member as usize == plane)?;
                    let prefix = planes[..position]
                        .iter()
                        .map(|member| u64::from(self.packet.planes[*member as usize].bytes_per_group))
                        .sum::<u64>();
                    Some((index, prefix))
                }
                RowPlaneContent::Codes { .. } => None,
            })
            .expect("every packet plane has one row plane");
        let group_bytes = u64::from(self.planes[index].bytes_per_group);
        (row * geometry.stride + geometry.offsets[index] + packet * group_bytes + prefix) * 8
            + u64::from(bit)
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
            usize::try_from(self.bytes(extents).expect("row layout tensor has a canonical byte count"))
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
        let (&extent, leading) = extents.split_last().expect("row layout tensor has a packing axis");
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
        let (&extent, leading) = extents.split_last().expect("row layout tensor has a packing axis");
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
/// The registered conversion of external `source` into resident storage in
/// `layout`: the resident form of an external representation for one layout.
/// `None` for dense and packed sources, and for a layout the resident
/// representation is not registered in. The table build asserts at most one
/// conversion per (source, representation, layout).
pub fn resident_conversion(
    source: RepresentationId,
    layout: Layout,
) -> Option<&'static RepresentationConversion> {
    internals::resident_conversion(source, layout)
}
/// The storage of packed representation `representation` (a representation
/// name such as `q4k`) in `layout`. Dense storage exists only in `packet`.
pub fn storage(representation: &str, layout: Layout) -> Option<RepresentationId> {
    representations()
        .iter()
        .find(|info| info.representation == representation && info.layout == layout)
        .map(|info| info.id)
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
            .try_fold(u64::from(dtype.bytes()), |bytes, extent| bytes.checked_mul(*extent)),
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
    use crate::repr::{PlaneEncoding, Repr, REPRS};
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
                representation: dtype.name(),
                layout: Layout::Packet,
                kind: RepresentationKind::Dense(dtype),
                decoded: dtype,
                access: RepresentationAccess::ReadWrite,
            });
        }
        for repr in REPRS {
            representations.push(RepresentationInfo {
                id: RepresentationId::new(representations.len() as u32),
                name: repr.name,
                representation: repr.name,
                layout: Layout::Packet,
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
                representation: name,
                layout: Layout::Packet,
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
        // Row layouts exist for every resident representation of an external
        // source: the resident forms the engine imports.
        for (_, _, _, resident) in external_specs {
            let repr = REPRS
                .iter()
                .find(|repr| repr.name == resident)
                .expect("external conversion names a registered packed representation");
            for layout in [Layout::Rows16, Layout::Mma16] {
                let name: &'static str =
                    Box::leak(format!("{resident}@{}", layout.as_str()).into_boxed_str());
                representations.push(RepresentationInfo {
                    id: RepresentationId::new(representations.len() as u32),
                    name,
                    representation: repr.name,
                    layout,
                    kind: RepresentationKind::PackedRows(row_layout(repr, layout)),
                    decoded: DType::F32,
                    access: RepresentationAccess::ReadOnly,
                });
            }
        }
        let mut conversions: Vec<RepresentationConversion> = Vec::new();
        for (source_name, _, _, destination_name) in external_specs {
            let source = representations
                .iter()
                .find(|representation| representation.name == source_name)
                .expect("registered external representation is present")
                .id;
            for destination in representations
                .iter()
                .filter(|representation| representation.representation == destination_name)
            {
                let (packet, kind) = match &destination.kind {
                    RepresentationKind::Packed(packet) => (packet, ConversionKind::Packet),
                    RepresentationKind::PackedRows(rows) => (
                        &rows.packet,
                        match rows.layout {
                            Layout::Rows16 => ConversionKind::Row,
                            Layout::Mma16 => ConversionKind::RowTile {
                                rows: MMA_TILE_ROWS as u32,
                            },
                            Layout::Packet => unreachable!("a row layout is never `packet`"),
                        },
                    ),
                    RepresentationKind::Dense(_) | RepresentationKind::External(_) => {
                        panic!("external conversion destination is not packed")
                    }
                };
                let recipe = external_repack_recipe(source_name, packet);
                validate_repack_recipe(
                    representations[source.index()].kind.clone(),
                    packet,
                    &recipe,
                );
                assert!(
                    conversions.iter().all(|conversion| {
                        let existing = &representations[conversion.destination.index()];
                        conversion.source != source
                            || existing.representation != destination.representation
                            || existing.layout != destination.layout
                    }),
                    "external representation `{source_name}` has more than one conversion into `{}`",
                    destination.name
                );
                conversions.push(RepresentationConversion {
                    id: RepresentationConversionId::new(conversions.len() as u32),
                    source,
                    destination: destination.id,
                    recipe,
                    kind,
                });
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

    /// The row layout of one packed representation. Planes, in order:
    /// 1. code planes from the packet `words` plane: 4-bit codes -> `codes_lo`;
    ///    5- and 6-bit codes -> `codes_lo` (low 4 bits) and `codes_hi` (the
    ///    remaining 1 or 2 bits); 8-bit codes -> `codes`;
    /// 2. `scales`: the hierarchical local coefficients exactly as the packet
    ///    `coefficients` plane packs them (never widened);
    /// 3. `supers`: every remaining coefficient plane of one packet
    ///    concatenated per group (k-quant super factors `d, dmin`; a direct
    ///    per-group scale).
    fn row_layout(repr: &Repr, layout: Layout) -> PackedRowLayout {
        let packet = packed_layout(repr);
        let words = &packet.planes[0];
        assert_eq!(words.name, "words", "packet plane 0 holds the codes");
        let group = packet.group;
        let code_plane = |name, shift, bits: u32| RowPlaneInfo {
            name,
            bytes_per_group: group * bits / 8,
            content: RowPlaneContent::Codes { shift, bits },
        };
        let mut planes = match words.entry_bits {
            4 => vec![code_plane("codes_lo", 0, 4)],
            5 | 6 => vec![
                code_plane("codes_lo", 0, 4),
                code_plane("codes_hi", 4, words.entry_bits - 4),
            ],
            8 => vec![code_plane("codes", 0, 8)],
            bits => panic!("`{}` has no row layout for {bits}-bit codes", repr.name),
        };
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
        if !scales.is_empty() {
            planes.push(RowPlaneInfo {
                name: "scales",
                bytes_per_group: group_bytes(&scales),
                content: RowPlaneContent::Groups { planes: scales },
            });
        }
        let supers = coefficients(&["scale_factor", "bias_factor", "scale", "bias", "block_scale"]);
        assert!(!supers.is_empty(), "`{}` has no super coefficients", repr.name);
        planes.push(RowPlaneInfo {
            name: "supers",
            bytes_per_group: group_bytes(&supers),
            content: RowPlaneContent::Groups { planes: supers },
        });
        let mut covered: Vec<u32> = std::iter::once(0)
            .chain(planes.iter().flat_map(|plane| match &plane.content {
                RowPlaneContent::Groups { planes } => planes.clone(),
                RowPlaneContent::Codes { .. } => Vec::new(),
            }))
            .collect();
        covered.sort_unstable();
        assert!(
            covered.iter().copied().eq(0..packet.planes.len() as u32),
            "row layout of `{}` covers every packet plane exactly once",
            repr.name
        );
        let group_multiple = match layout {
            Layout::Rows16 => 1,
            Layout::Mma16 => {
                let block = MMA_KBLOCK as u32;
                assert!(
                    block.is_multiple_of(group) || group.is_multiple_of(block),
                    "`{}` groups do not tile 64-column k-blocks",
                    repr.name
                );
                (block / group).max(1)
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
        layout: Layout,
    ) -> Option<&'static RepresentationConversion> {
        tables().conversions.iter().find(|conversion| {
            conversion.source == source
                && representation_info(conversion.destination).layout == layout
        })
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
    fn resident_conversions_are_keyed_by_source_representation_and_layout() {
        let q4_k = representation("gguf_q4_k").unwrap();
        for (layout, name, kind) in [
            (Layout::Packet, "q4k", ConversionKind::Packet),
            (Layout::Rows16, "q4k@rows16", ConversionKind::Row),
            (Layout::Mma16, "q4k@mma16", ConversionKind::RowTile { rows: 16 }),
        ] {
            let conversion = resident_conversion(q4_k, layout).unwrap();
            assert_eq!(conversion.destination, representation(name).unwrap());
            assert_eq!(storage("q4k", layout), Some(conversion.destination));
            assert_eq!(conversion.kind, kind);
        }
        for info in representations() {
            for layout in Layout::ALL {
                let conversion = resident_conversion(info.id, layout);
                match info.kind {
                    RepresentationKind::External(_) => {
                        let conversion = conversion.unwrap();
                        assert_eq!(conversion.source, info.id);
                        assert_eq!(representation_info(conversion.destination).layout, layout);
                        assert_eq!(
                            representation_conversion(info.id, conversion.destination),
                            Some(conversion)
                        );
                    }
                    RepresentationKind::Dense(_)
                    | RepresentationKind::Packed(_)
                    | RepresentationKind::PackedRows(_) => assert_eq!(conversion, None),
                }
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
            [("codes_lo", 128), ("codes_hi", 32), ("scales", 12), ("supers", 4)]
        );
        assert_eq!(
            plane_set("q6k@rows16"),
            [("codes_lo", 128), ("codes_hi", 64), ("scales", 16), ("supers", 2)]
        );
        assert_eq!(plane_set("q8g32s@rows16"), [("codes", 32), ("supers", 2)]);
        assert_eq!(plane_set("iq4g32@rows16"), [("codes_lo", 128), ("supers", 32)]);
        // Qwen3.5 K = 2560: codes 1280 | scales 120 -> 128 | supers 40 -> 48.
        let q4k = row_layout("q4k@rows16");
        assert_eq!(q4k.plane_row_offset(1, 2560), Some(1280));
        assert_eq!(q4k.plane_row_offset(2, 2560), Some(1408));
        assert_eq!(q4k.row_stride_bytes(2560), Some(1456));
        assert_eq!(canonical_bytes(representation("q4k@rows16").unwrap(), &[3, 2560]), Some(3 * 1456));
        // A q8 row of three groups: 96 code bytes, 6 super bytes -> 16.
        let q8 = row_layout("q8g32s@rows16");
        assert_eq!(q8.row_stride_bytes(96), Some(112));
        // mma16 pads rows per matrix and q8 rows to whole 64-column k-blocks.
        let q8_mma = representation("q8g32s@mma16").unwrap();
        assert_eq!(row_layout("q8g32s@mma16").row_stride_bytes(96), Some(128 + 16));
        assert_eq!(canonical_bytes(q8_mma, &[2, 17, 96]), Some(2 * 32 * 144));
        assert_eq!(canonical_bytes(q8_mma, &[96]), None);
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
                            let column = 64 * block + 16 * step + 2 * t + 8 * (register >> 1) + half;
                            let v = (block * 32 + lane) * 128 + 32 * step + 4 * (register + 4 * half);
                            let expected = (v / 8 / row_bytes * stride + v / 8 % row_bytes) * 8 + v % 8;
                            assert_eq!(layout.code_bit(&geometry, 0, row, column), expected);
                        }
                    }
                }
            }
        }
    }

    /// Deterministic canonical bytes of an external tensor.
    fn source_bytes(source: RepresentationId, shape: &[u64], seed: u32) -> Vec<u8> {
        let length = canonical_bytes(source, shape).unwrap() as usize;
        (0..length)
            .map(|index| {
                let x = (index as u32).wrapping_add(seed).wrapping_mul(2_654_435_761);
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
        use crate::interp::{repack, TensorData};
        for source_info in representations()
            .iter()
            .filter(|info| matches!(info.kind, RepresentationKind::External(_)))
        {
            let RepresentationKind::External(external) = &source_info.kind else {
                unreachable!()
            };
            let group = u64::from(external.logical_group);
            for shape in [vec![17, 3 * group], vec![2, 3, 2 * group - 8], vec![1, group]] {
                let host_shape = shape.iter().map(|extent| *extent as usize).collect::<Vec<_>>();
                let bytes = source_bytes(source_info.id, &shape, shape[0] as u32);
                let decode = |conversion: &RepresentationConversion| {
                    let converted = repack(conversion.id, &host_shape, &bytes).unwrap();
                    let data =
                        TensorData::encoded(conversion.destination, host_shape.clone(), converted.clone())
                            .unwrap();
                    (converted, data.values().unwrap())
                };
                let packet = resident_conversion(source_info.id, Layout::Packet).unwrap();
                let (packet_bytes, expected) = decode(packet);
                for layout in [Layout::Rows16, Layout::Mma16] {
                    let conversion = resident_conversion(source_info.id, layout).unwrap();
                    let (converted, actual) = decode(conversion);
                    let RepresentationKind::PackedRows(rows) =
                        &representation_info(conversion.destination).kind
                    else {
                        unreachable!()
                    };
                    assert_eq!(rows.packets(&shape, &converted), packet_bytes);
                    assert_eq!(rows.place(&shape, &packet_bytes), converted);
                    assert_eq!(
                        actual.iter().map(|value| value.to_bits()).collect::<Vec<_>>(),
                        expected.iter().map(|value| value.to_bits()).collect::<Vec<_>>(),
                        "{} -> {} over {shape:?}",
                        source_info.name,
                        representation_info(conversion.destination).name
                    );
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
