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
    BiasKind, CodeInterpretation, CodeKind, CodingAxes, CoefficientKind, DecodeAxes, DecodeComponent,
    DecodeRecipe, DecodeStep, DecodeTemp, FloatCodeFormat, GridCodes, GridSigns, LocalBias,
    LocalCodes, LookupTable, MinimumSign, OperandRecipe, PlaneEncoding, PlaneField, PlaneRead,
    PlaneSchema, QuantizedComponent, ScaleEncoding, ScaledCodes, ValueCodes, GRID_VALUES,
};

/// A weight's arithmetic (`DecodeAxes::profile`), what consumer forms state
/// their requirements over.
pub mod profile {
    pub use crate::repr::{Bias, Domain, Local, LocalScale, Profile, Upper, ValueMap};
}

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

    /// Whether its runtime encodes each run's launches on the host, so a
    /// launch can be left out by a word the host reads then (`launch …
    /// if (w)`, `unless (w)`). CUDA and Vulkan replay the graph of the
    /// launches a submission encodes, keyed by them.
    pub const fn encodes_runs(self) -> bool {
        match self {
            BackendName::Metal | BackendName::Cuda | BackendName::Vulkan => true,
            BackendName::Cpu => false,
        }
    }

    /// Whether the backend's devices see the host's writes while work runs,
    /// so a call that parks (an awaited port, `&await`) waits inside its
    /// kernel, which spins on the awaited words. Where they do not, the
    /// runtime parks the call at a submission boundary: the work before it
    /// ends its submission and signals the host, and the call's submission
    /// waits for the host's answer (`NativeGraphDraft::enqueue`). A Vulkan
    /// device's running work neither sees a host write nor makes its own
    /// visible to the host before it ends (the Vulkan spec has no host domain
    /// operation within a submission; measured on GB10, and on an RTX 3090
    /// outside its BAR).
    pub const fn parks_in_kernels(self) -> bool {
        match self {
            BackendName::Cpu | BackendName::Metal | BackendName::Cuda => true,
            BackendName::Vulkan => false,
        }
    }

    /// Whether the backend's plain device loads read what the host last
    /// wrote of the records a record parameter binds (`records`), which the
    /// host lands while work that reads them runs (a bank's records, a
    /// tier's, a conveyor's). Where they do not, the generated prefix marks
    /// each record parameter host-written (`<PORT>_HOST_WRITTEN`), and the
    /// backend's native library reads a host-written port's records with
    /// loads that do. Metal's plain loads return stale bytes of a record the
    /// host rewrites and the device reads again within one command buffer,
    /// which neither a system-scope fence nor a kernel boundary prevents;
    /// `volatile coherent(system)` loads are exact, at 8% of a whole
    /// `routed_down` launch on an M4 Pro, which a launch that runs while
    /// nothing lands in its records avoids. A CUDA device whose
    /// memory is the host's (GB10) is coherent with the host's writes, as
    /// measured on that device, and a Vulkan device reads a host write only
    /// across a submission boundary, where a call that waits for one parks
    /// (`parks_in_kernels`), so their plain loads do. The CPU's loads are the
    /// host's.
    pub const fn plain_loads_read_host_writes(self) -> bool {
        match self {
            BackendName::Cpu | BackendName::Cuda | BackendName::Vulkan => true,
            BackendName::Metal => false,
        }
    }

    /// Whether a recorded graph is updated to another run's storage in
    /// queue order beside its launches in flight, so one graph serves every
    /// run of its shape: Vulkan rewrites argument blocks on the queue, and a
    /// Metal schedule binds nothing. CUDA sets a graph's parameters on the
    /// host, which waits behind its launches in flight, so a CUDA graph is
    /// updated only once idle and every run in flight keeps its own.
    pub const fn graphs_update_in_flight(self) -> bool {
        match self {
            BackendName::Cuda => false,
            BackendName::Cpu | BackendName::Metal | BackendName::Vulkan => true,
        }
    }

    /// Whether the backend runs fill work (a native submission queued on the
    /// fill queue, `NativeQueue::Fill`) on a device queue of its own beside
    /// compute, so a fill's moves and its parks overlap compute and hold
    /// only later fills. CUDA runs it on a second native stream. Vulkan (a
    /// second queue with timeline semaphores) and Metal (a second command
    /// queue with a shared event) have none yet, so their fill queue is the
    /// compute queue; the CPU's work completes as it is submitted.
    pub const fn fills_beside_compute(self) -> bool {
        match self {
            BackendName::Cuda => true,
            BackendName::Cpu | BackendName::Metal | BackendName::Vulkan => false,
        }
    }

    /// Whether a sampled graph run stamps the device's time at its plan's
    /// marks (`NativeSpans`): CUDA with events and Vulkan with timestamps.
    /// Metal and the CPU have no span timing.
    pub const fn times_spans(self) -> bool {
        match self {
            BackendName::Cuda | BackendName::Vulkan => true,
            BackendName::Cpu | BackendName::Metal => false,
        }
    }

    /// Whether the backend runs a resident launch (`launch K resident
    /// (BARRIER)`): every threadgroup of its grid on the device at once, so
    /// they may wait for one another at grid barriers. Metal sizes the grid
    /// from the formed pipeline and the device's cores (`RESIDENT`), and
    /// every barrier wait is bounded: nothing guarantees co-residency, so a
    /// give-up fails the submission instead of hanging it. CUDA would take a
    /// cooperative launch, which no runtime path issues yet; Vulkan has
    /// neither an occupancy query nor a cooperative launch; the CPU runs
    /// groups as tasks a worker finishes one at a time.
    pub const fn resident_launches(self) -> bool {
        matches!(self, BackendName::Metal)
    }

    /// Whether a native implementation dispatches a block of its launches
    /// more than once (`repeat`): its runtime's encoder replays the block.
    pub const fn repeats_launches(self) -> bool {
        match self {
            BackendName::Metal => true,
            BackendName::Cpu | BackendName::Cuda | BackendName::Vulkan => false,
        }
    }

    /// Whether a launch with a written grid (`grid (w)`) is authored as the
    /// body of one block ([`written_grid_block`]), which the runtime's
    /// generated kernel runs over the grid's blocks: CUDA has no indirect
    /// launch. Elsewhere the launch is the kernel, dispatched indirectly.
    pub const fn authors_written_grid_blocks(self) -> bool {
        match self {
            BackendName::Cuda => true,
            BackendName::Cpu | BackendName::Metal | BackendName::Vulkan => false,
        }
    }

    /// Whether a native implementation reaches tensors a record at a time
    /// through a table of record addresses (`records`): its runtime binds
    /// such a parameter as that table, which reserved storage held as
    /// chunks requires.
    pub const fn binds_record_tables(self) -> bool {
        match self {
            BackendName::Metal | BackendName::Vulkan => true,
            BackendName::Cpu | BackendName::Cuda => false,
        }
    }
}

/// The bytes of a resident launch's barrier, at the start of its `sync`
/// scratch buffer: four words, arrivals, departures, the give-up word and
/// a pad (`seismic_grid` in the runtime's `native/metal_resident.h`).
pub const RESIDENT_BARRIER_BYTES: u64 = 16;

/// The byte of a resident barrier's give-up word, which the runtime reads
/// once the work completes (`NativeSubmission::wait`).
pub const RESIDENT_GIVE_UP_BYTE: u64 = 8;

/// The bytes of one entry of a record table: the 8-byte device address of
/// its record.
pub const RECORD_TABLE_ENTRY_BYTES: u64 = 8;

/// The bytes of a record table of `records` entries, one device address a
/// record; `None` past the bytes a `u64` counts.
pub const fn record_table_bytes(records: u64) -> Option<u64> {
    records.checked_mul(RECORD_TABLE_ENTRY_BYTES)
}

/// Where entry `index` of a formed record table lies, in bytes from its
/// first: an entry of a table whose bytes [`record_table_bytes`] counted,
/// so it lies within them.
pub const fn record_table_entry(index: u64) -> u64 {
    index * RECORD_TABLE_ENTRY_BYTES
}

/// The bytes of a record table whose entries are `addresses` in turn, each
/// a record's device address, little-endian.
pub fn record_table(addresses: impl IntoIterator<Item = u64>) -> Vec<u8> {
    addresses
        .into_iter()
        .flat_map(u64::to_le_bytes)
        .collect()
}

/// The table entries of chunk `index` of `granule` bytes, at device address
/// `address`, in a reservation of `records` records of `record` bytes: its
/// first record, and the table bytes of its records' addresses, each
/// record's within the chunk (the last chunk may hold fewer).
pub fn chunk_entries(index: u64, granule: u64, record: u64, records: u64, address: u64) -> (u64, Vec<u8>) {
    let per = granule / record;
    let start = (index * per).min(records);
    let end = ((index + 1) * per).min(records);
    (start, record_table((start..end).map(|entry| address + (entry - index * per) * record)))
}

/// The function a written-grid launch `kernel` is authored as where its
/// backend authors written grids as blocks
/// ([`BackendName::authors_written_grid_blocks`]): the body of one block.
pub fn written_grid_block(kernel: &str) -> String {
    format!("{kernel}_block")
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
/// every cache identity. Registering a new entry changes no other entry's
/// meaning nor its identity (`ids::registry_id`, keyed by what names it), so
/// it changes no revision.
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
/// - `Rows8`: eight-row tiles, each plane a span per storage group holding
///   its eight rows' shares: a code plane in panel order (32-byte units of
///   `[row][4 bytes]`, slot `s` of byte `j` a row's code of column `4s + j`
///   of the unit's columns), so one aligned load is a step of four columns
///   of the tile's rows; a group plane record-major (each record, a
///   coefficient group's fields, of the eight rows in turn); a grid plane
///   its rows' shares in turn. The CPU resident layout (`code_bit`,
///   `packet_bit`).
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
    /// Where a packet's bytes lie: `None` one after another, a row's packets
    /// in turn; under a companion grid its two planes.
    pub planes: Option<GridPlanes>,
}

/// The planes of a source under a companion grid of scales, as a file
/// stores the codes and their grid: each matrix's codes, a row's after the
/// row before, then its grid of one scale per `tile` of values, a tile row's
/// after the one before. A packet (its scale's `scale_bytes`, then its
/// group's codes) gathers its scale from the grid and its codes from the
/// codes plane, which [`ExternalPacketLayout::packet_byte`] addresses.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GridPlanes {
    pub scale_bytes: u32,
    /// `[rows, columns]` of values each scale scales: a whole number of
    /// packets along a row.
    pub tile: [u32; 2],
}

/// Where a packet's bytes start within its matrix ([`ExternalPacketLayout::
/// packet_pieces`]): its codes, and under a grid its scale apart.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PacketPieces {
    /// The packet's first byte, or under a grid its first code's.
    pub codes: u64,
    /// Its scale's first byte, under a grid.
    pub scale: Option<u64>,
}

impl ExternalPacketLayout {
    /// The bytes a packet's codes take: all of it, or under a grid all but
    /// its scale.
    pub const fn code_bytes(&self) -> u32 {
        match self.planes {
            None => self.packet_size,
            Some(planes) => self.packet_size - planes.scale_bytes,
        }
    }

    /// The bytes of one `[rows, columns]` matrix: its rows' packets, and
    /// under a grid its grid of scales after them (whole tiles, a partial
    /// last one counted whole). `None` on overflow.
    pub fn matrix_bytes(&self, rows: u64, columns: u64) -> Option<u64> {
        let packets = rows.checked_mul(columns.div_ceil(u64::from(self.logical_group)))?;
        let codes = packets.checked_mul(u64::from(self.code_bytes()))?;
        match self.planes {
            None => Some(codes),
            Some(planes) => rows
                .div_ceil(u64::from(planes.tile[0]))
                .checked_mul(columns.div_ceil(u64::from(planes.tile[1])))?
                .checked_mul(u64::from(planes.scale_bytes))?
                .checked_add(codes),
        }
    }

    /// Where packet `packet` of row `row` of a `[rows, columns]` matrix
    /// lies, from the matrix's first byte: the addressing of a source's
    /// packets the interpreter and the CPU's conversions take (through
    /// [`SourceMatrices`]). The devices' conversion code and streaming
    /// windows lay and read the same planes; their tests compare their bytes
    /// with the host's conversion.
    pub fn packet_pieces(&self, rows: u64, columns: u64, row: u64, packet: u64) -> PacketPieces {
        let row_packets = columns.div_ceil(u64::from(self.logical_group));
        let codes = (row * row_packets + packet) * u64::from(self.code_bytes());
        PacketPieces {
            codes,
            scale: self.planes.map(|planes| {
                let [tile_rows, tile_columns] = planes.tile.map(u64::from);
                let tile = row / tile_rows * columns.div_ceil(tile_columns)
                    + packet * u64::from(self.logical_group) / tile_columns;
                rows * row_packets * u64::from(self.code_bytes()) + tile * u64::from(planes.scale_bytes)
            }),
        }
    }

    /// Where byte `byte` of packet `packet` of row `row` of a `[rows,
    /// columns]` matrix lies, from the matrix's first byte
    /// ([`Self::packet_pieces`]): a packet's bytes in order, under a grid
    /// its scale's then its codes'.
    pub fn packet_byte(&self, rows: u64, columns: u64, row: u64, packet: u64, byte: u32) -> u64 {
        let pieces = self.packet_pieces(rows, columns, row, packet);
        match (self.planes, pieces.scale) {
            (Some(planes), Some(scale)) if byte < planes.scale_bytes => scale + u64::from(byte),
            (Some(planes), _) => pieces.codes + u64::from(byte - planes.scale_bytes),
            (None, _) => pieces.codes + u64::from(byte),
        }
    }
}

/// A source tensor as its packets lie ([`ExternalPacketLayout::matrices`]):
/// `count` matrices of `rows` rows of `columns` values, each `bytes` bytes
/// after the one before. A source of one plane is one matrix of all its
/// rows (its packets one after another, whatever its axes); under a grid
/// every axis before the last two counts matrices, and a tensor of one axis
/// is one row.
#[derive(Clone, Copy, Debug)]
pub struct SourceMatrices<'a> {
    pub layout: &'a ExternalPacketLayout,
    pub count: u64,
    pub rows: u64,
    pub columns: u64,
    pub bytes: u64,
}

impl ExternalPacketLayout {
    /// The matrices of a tensor of `extents` of this source; `None` for a
    /// tensor of no axes, or on overflow.
    pub fn matrices(&self, extents: &[u64]) -> Option<SourceMatrices<'_>> {
        let (&columns, leading) = extents.split_last()?;
        let product = |extents: &[u64]| extents.iter().try_fold(1u64, |product, extent| product.checked_mul(*extent));
        let (count, rows) = match self.planes {
            None => (1, product(leading)?),
            // A matrix has rows: fewer than two axes is no planar tensor.
            Some(_) => {
                let (rows, matrices) = leading.split_last()?;
                (product(matrices)?, *rows)
            }
        };
        let bytes = self.matrix_bytes(rows, columns)?;
        count.checked_mul(bytes)?;
        count.checked_mul(rows)?;
        Some(SourceMatrices { layout: self, count, rows, columns, bytes })
    }
}

impl SourceMatrices<'_> {
    /// Packets of each row.
    pub fn row_packets(&self) -> u64 {
        self.columns.div_ceil(u64::from(self.layout.logical_group))
    }

    /// Rows of the tensor, every matrix's.
    pub fn tensor_rows(&self) -> u64 {
        self.count * self.rows
    }

    /// Bytes of the tensor.
    pub fn tensor_bytes(&self) -> u64 {
        self.count * self.bytes
    }

    /// Where packet `packet` of row `row` of the tensor (its matrices' rows
    /// in turn) lies from the tensor's first byte: its matrix's, then
    /// [`ExternalPacketLayout::packet_pieces`] within it.
    pub fn packet_pieces(&self, row: u64, packet: u64) -> PacketPieces {
        let matrix = row / self.rows * self.bytes;
        let pieces = self.layout.packet_pieces(self.rows, self.columns, row % self.rows, packet);
        PacketPieces { codes: matrix + pieces.codes, scale: pieces.scale.map(|scale| matrix + scale) }
    }

    /// The bytes packets `packets` (not none) of row `row` of the tensor
    /// take, from its first byte ([`Self::packet_pieces`]): their packets',
    /// or under a grid their codes', then their scales'.
    pub fn packet_spans(
        &self,
        row: u64,
        packets: std::ops::Range<u64>,
    ) -> impl Iterator<Item = std::ops::Range<u64>> + Clone {
        assert!(!packets.is_empty(), "a span of packets holds one");
        let (first, last) = (self.packet_pieces(row, packets.start), self.packet_pieces(row, packets.end - 1));
        let code_bytes = self.layout.code_bytes();
        let scale_bytes = u64::from(self.layout.packet_size - code_bytes);
        let scales = first.scale.zip(last.scale).map(|(first, last)| first..last + scale_bytes);
        std::iter::once(first.codes..last.codes + u64::from(code_bytes)).chain(scales)
    }

    /// The tensor's packets of `bytes`, its stored bytes, one after another
    /// (rows in turn, a row's packets in turn), each its bytes in order:
    /// `bytes` itself for a source of one plane, gathered from its pieces
    /// under a grid. Panics unless `bytes` is the tensor's.
    pub fn packets<'b>(&self, bytes: &'b [u8]) -> std::borrow::Cow<'b, [u8]> {
        assert_eq!(bytes.len() as u64, self.tensor_bytes(), "a source tensor's bytes");
        if self.layout.planes.is_none() {
            return std::borrow::Cow::Borrowed(bytes);
        }
        let (code_bytes, packet_size) = (self.layout.code_bytes() as usize, self.layout.packet_size as usize);
        let mut packets = Vec::with_capacity((self.tensor_rows() * self.row_packets()) as usize * packet_size);
        for row in 0..self.tensor_rows() {
            for packet in 0..self.row_packets() {
                let pieces = self.packet_pieces(row, packet);
                let scale = pieces.scale.expect("a packet under a grid has a scale") as usize;
                let codes = pieces.codes as usize;
                packets.extend_from_slice(&bytes[scale..scale + packet_size - code_bytes]);
                packets.extend_from_slice(&bytes[codes..codes + code_bytes]);
            }
        }
        std::borrow::Cow::Owned(packets)
    }

    /// Byte `byte` of every packet of the tensor, from its first byte
    /// ([`ExternalPacketLayout::packet_byte`]): under a grid a scale's
    /// bytes once for each packet it scales.
    fn packet_bytes(&self, byte: u32) -> impl Iterator<Item = u64> + '_ {
        (0..self.tensor_rows()).flat_map(move |row| {
            let matrix = row / self.rows * self.bytes;
            (0..self.row_packets()).map(move |packet| {
                matrix + self.layout.packet_byte(self.rows, self.columns, row % self.rows, packet, byte)
            })
        })
    }
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
    /// The layout of packed representation kind `kind`; `None` for another.
    pub fn of(kind: &'static RepresentationKind) -> Option<Self> {
        match kind {
            RepresentationKind::Packed(packet) => Some(Self::Packet(packet)),
            RepresentationKind::PackedRows(rows) => Some(Self::Rows(rows)),
            RepresentationKind::Dense(_) | RepresentationKind::External(_) => None,
        }
    }

    /// The destination's packet form.
    pub fn packet(self) -> &'static PackedPacketLayout {
        match self {
            Self::Packet(packet) => packet,
            Self::Rows(rows) => &rows.packet,
        }
    }
}

impl RepresentationConversion {
    /// Brings `source`, the arbitrary bytes of a source tensor of `extents`,
    /// into the conversion's domain, changing as few bits as that takes.
    /// The domain is the source packets whose binary16 fields the recipe
    /// converts as numbers ([`PacketRepackRecipe::converted_f16_fields`])
    /// are finite: a NaN's payload is not part of the contract, since a
    /// device's conversion quiets a signaling NaN where the host reference
    /// widens its bits. Each such field has the high bit of its exponent
    /// cleared. Panics unless `source` is the tensor's canonical bytes.
    pub fn admit_source(&self, extents: &[u64], source: &mut [u8]) {
        let bits = self.recipe.converted_f16_fields().iter().map(|field| field + 14).collect::<Vec<_>>();
        self.clear(extents, source, &bits);
    }

    /// Brings `source`, the arbitrary bytes of a source tensor of
    /// `extents`, into the conversion's domain ([`Self::admit_source`]) with
    /// every number the destination reads from them finite: also each
    /// floating-point field it carries bit for bit into a plane of
    /// floating-point numbers (a GGUF scale it keeps as stored) has the high
    /// bit of its exponent cleared, and each E4M3 code that is NaN is made
    /// finite ([`PlaneEncoding::nan_pattern`]); every finite code is kept.
    /// A record of such packets has a finite value everywhere, so arithmetic
    /// over it compares; one of a carried NaN or infinity does not.
    pub fn finite_source(&self, extents: &[u64], source: &mut [u8]) {
        let mut bits = self.recipe.converted_f16_fields().iter().map(|field| field + 14).collect::<Vec<_>>();
        for (plane, recipe) in self.destination_layout.packet().planes.iter().zip(&self.recipe.planes) {
            // A typed recipe's numbers are converted (above) or a
            // codebook's, never carried bits.
            if let (Some(high), PlaneRepackRecipe::BitRoutes(routes)) = (plane.encoding.finite_bit(), recipe) {
                bits.extend(
                    routes
                        .chunks_exact(plane.encoding.entry_bits() as usize)
                        .filter_map(|entry| entry[high as usize]),
                );
            }
        }
        let mut nans = Vec::new();
        for (plane, recipe) in self.destination_layout.packet().planes.iter().zip(&self.recipe.planes) {
            if let (Some((mask, clear)), PlaneRepackRecipe::BitRoutes(routes)) = (plane.encoding.nan_pattern(), recipe) {
                for entry in routes.chunks_exact(plane.encoding.entry_bits() as usize) {
                    let pattern = (0..entry.len() as u32).filter(|bit| mask >> bit & 1 == 1).map(|bit| entry[bit as usize]).collect::<Option<Vec<_>>>();
                    if let (Some(pattern), Some(clear)) = (pattern, entry[clear as usize]) {
                        nans.push((pattern, clear));
                    }
                }
            }
        }
        self.clear(extents, source, &bits);
        self.clear_nans(extents, source, &nans);
    }

    /// Each pattern of packet bits `nans` (its bits, and the one that
    /// clearing leaves any other entry's bits alone) wherever all of its
    /// bits are set in a packet of `source`, the bytes of a source tensor
    /// of `extents`, has that one bit cleared: a NaN code made finite and no
    /// finite code changed.
    fn clear_nans(&self, extents: &[u64], source: &mut [u8], nans: &[(Vec<u32>, u32)]) {
        let matrices = self.source_layout.matrices(extents).expect("a source tensor's extents");
        for (pattern, clear) in nans {
            let mut bits = pattern.iter().map(|&bit| matrices.packet_bytes(bit / 8)).collect::<Vec<_>>();
            let mut cleared = matrices.packet_bytes(clear / 8);
            loop {
                let at = bits.iter_mut().map(Iterator::next).collect::<Option<Vec<_>>>();
                let (Some(at), Some(cleared)) = (at, cleared.next()) else {
                    break;
                };
                if pattern.iter().zip(&at).all(|(bit, at)| source[*at as usize] >> (bit % 8) & 1 == 1) {
                    source[cleared as usize] &= !(1 << (clear % 8));
                }
            }
        }
    }

    /// Clear packet bits `bits` of every packet of `source`, the bytes of a
    /// source tensor of `extents`, where each lies
    /// ([`SourceMatrices::packet_bytes`]).
    fn clear(&self, extents: &[u64], source: &mut [u8], bits: &[u32]) {
        let matrices = self.source_layout.matrices(extents).expect("a source tensor's extents");
        assert_eq!(source.len() as u64, matrices.tensor_bytes(), "a source tensor's bytes");
        for &bit in bits {
            for at in matrices.packet_bytes(bit / 8) {
                source[at as usize] &= !(1 << (bit % 8));
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
                RepackExpr::ScaleF32 { value, .. } => collect(value, fields),
                RepackExpr::Lookup { index, .. } => collect(index, fields),
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
    /// For every destination plane bit, the source packet bit ordinal it
    /// moves, or none: a zero bit.
    BitRoutes(Vec<Option<u32>>),
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
            PlaneEncoding::Packed { .. }
            | PlaneEncoding::Grid(_)
            | PlaneEncoding::SignedLocal { .. } => 4,
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

    /// Byte offset of a plane within packet `packet`.
    pub fn plane_offset(&self, packet: u64, plane: usize) -> Option<u64> {
        packet
            .checked_mul(u64::from(self.packet_size))?
            .checked_add(u64::from(self.planes.get(plane)?.offset))
    }

    /// Canonical packets of `rows` rows of `logical_extent` elements, assembled
    /// from plane-separated bytes. `planes` names every plane of this layout
    /// exactly once, in any order. Plane `p` supplies
    /// `rows × packet_extent(logical_extent) × p.bytes_per_group` bytes ordered
    /// by (row, packet); each packet's slice is exactly the bytes that plane
    /// occupies inside one canonical packet. Packet bytes no plane occupies
    /// are zero.
    ///
    /// The caller has established that the canonical byte count of the
    /// tensor ([`canonical_bytes`]) fits `u64`; geometry that
    /// overflows is a caller contradiction and panics.
    pub fn packets_from_planes(
        &self,
        rows: u64,
        logical_extent: u64,
        planes: &[(&str, &[u8])],
    ) -> Result<Vec<u8>, PlaneAssemblyError> {
        let packet_count = self.packet_extent(logical_extent);
        let total = units_bytes(&[rows, packet_count], u64::from(self.packet_size))
            .expect("packed tensor geometry has a canonical byte count");
        let packets = rows
            .checked_mul(packet_count)
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

/// A source under a companion grid keeps its codes and its grid apart, so no
/// per-packet address reaches them: only its registered conversion reads
/// it, addressing both planes itself.
pub const PLANAR_SOURCE_IS_CONVERSION_ONLY: &str =
    "a source under a companion grid reached a per-packet address, which only its conversion may read";

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
/// Columns per `k16` fragment step of an `mma16` k-block.
pub const MMA_STEP: u64 = 16;
/// Fragment slots per `k16` step of an `mma16` lane.
pub const MMA_STEP_SLOTS: u64 = 8;
// `mma_fragment_byte` is written over these, in a kernel's terms.
const _: () = assert!(MMA_LANES == 32 && MMA_STEP_SLOTS == 8 && MMA_KBLOCK == 4 * MMA_STEP);

/// A placement rule, in one source: `$name` here, and `$text` the same
/// expression as the parameter list and body of a native ABI macro. Its
/// parameters are parenthesized for that macro, and its body is only integer
/// arithmetic every kernel dialect reads alike.
macro_rules! placement_rule {
    ($(#[$doc:meta])* $name:ident, $text:ident ($($parameter:ident),*) = $body:expr) => {
        $(#[$doc])*
        #[inline]
        #[allow(unused_parens)]
        pub const fn $name($($parameter: u64),*) -> u64 {
            $body
        }
        #[doc = concat!("[`Self::", stringify!($name), "`] as a native ABI macro's parameters and body.")]
        pub const $text: &'static str =
            concat!("(", stringify!($($parameter),*), ") (", stringify!($body), ")");
    };
}

/// The byte geometry of a packed representation in a row layout (`rows16`,
/// `mma16`). A tensor `[..., N, K]` is stored as rows of `K` logical values
/// (every leading coordinate is a row). Within a row the planes follow one
/// another in `planes` order, each starting 16 B-aligned; the row stride is
/// the padded sum, so every row starts 16 B-aligned. A row holds
/// `row_groups(K)` whole storage groups; bytes of padding groups are zero.
///
/// `mma16` additionally pads the row axis `N` of every `[N, K]` matrix to a
/// multiple of 16 rows (padding rows are zero) and permutes each code plane
/// within each 16-row tile (`code_bit`).
/// Scale and super planes, the row stride and every plane offset are exactly
/// those of `rows16`.
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
    /// The whole bytes of the vector-code entries of the packet `words`
    /// plane (one per `GRID_VALUES` columns), placed as a code plane of
    /// `bits` bits a column (`code_bit`): an entry is the codes of its
    /// columns. With `carried`, the entries also carry that packet plane, the
    /// representation's coefficient codes: bit `j` of a group's code at the
    /// grid's `carried_bit` of the group's entry `j` (in the grid high plane
    /// where that bit is past the entry's whole bytes). `grid` is the
    /// entries' grid.
    Grids {
        bits: u32,
        grid: &'static GridCodes,
        carried: Option<u32>,
    },
    /// Bits `[shift, shift + bits)` of every vector-code entry of the packet
    /// `words` plane, the part of an entry past its whole bytes (`grids`
    /// holds those), `bits` an entry in entry order: placed as a code plane
    /// of one unit an entry (`code_bit`).
    GridHigh { shift: u32, bits: u32 },
    /// The bytes of the listed packet planes of one packet, concatenated in
    /// list order, per storage group: the local or the super coefficients.
    Groups { level: CoefficientLevel, planes: Vec<u32> },
    /// `mma16` of a carried grid representation: the tile's whole storage
    /// groups as records of its 16 rows (`record_byte`), every group's
    /// carried entries in GEMV lane order, then every group's `factors`
    /// packet planes. The only plane of its layout. `grid` is the entries'
    /// grid.
    Records {
        bits: u32,
        grid: &'static GridCodes,
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
        logical_extent
            .div_ceil(u64::from(self.packet.group))
            .checked_next_multiple_of(u64::from(self.group_multiple))
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

    /// Bytes between the starts of adjacent rows; `None` on overflow or
    /// where a tile's rows would hold 2^32 bytes or more, since the native
    /// libraries address within a tile in 32 bits (CUDA's
    /// `packets::CodePlane`).
    pub fn row_stride_bytes(&self, logical_extent: u64) -> Option<u64> {
        let last = self.planes.len().checked_sub(1)?;
        let stride = align(
            self.plane_row_offset(last, logical_extent)?
                .checked_add(self.plane_bytes_per_row(last, logical_extent)?)?,
        )?;
        (stride.checked_mul(self.tile_rows())? <= u64::from(u32::MAX)).then_some(stride)
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
            *rows = rows.checked_next_multiple_of(self.tile_rows())?;
        }
        units.push(1);
        Some(units)
    }

    /// Canonical bytes of a tensor with `extents`: its storage units'
    /// bytes ([`units_bytes`]) at the row stride.
    pub fn bytes(&self, extents: &[u64]) -> Option<u64> {
        units_bytes(&self.storage_units(extents)?, self.row_stride_bytes(*extents.last()?)?)
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
    /// `rows8`: a codes plane's tile span of each storage group is units of
    /// `32 / bits` columns, `[row r][4 bytes]`, byte `j` of row `r` holding
    /// its codes of the unit's columns `4s + j` at bit `bits * s`; a grid
    /// plane's span holds its rows' shares in turn.
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
    /// `record_byte`. A `grid_high` plane is placed as a grids plane of one
    /// unit an entry (`GRID_VALUES` columns, `column` its first), `bits` a
    /// unit.
    pub fn code_bit(&self, geometry: &RowGeometry, plane: usize, row: u64, column: u64) -> u64 {
        // Bits of a placed unit, whether it is placed as entries are, and the
        // columns of a unit.
        let (bits, grids, unit) = match self.planes[plane].content {
            RowPlaneContent::Codes { bits, .. } => (bits, false, 1),
            RowPlaneContent::Grids { bits, .. } => (bits, true, 1),
            RowPlaneContent::GridHigh { bits, .. } => (bits, true, u64::from(GRID_VALUES)),
            RowPlaneContent::Records { bits, .. } => {
                let entry = column % u64::from(self.packet.group) / u64::from(GRID_VALUES);
                let packet = column / u64::from(self.packet.group);
                let byte = self.record_byte(geometry, row, packet, Record::Entries, |r| {
                    Self::record_entry(r, entry / 8, entry % 8)
                });
                return byte * 8 + column % u64::from(GRID_VALUES) * u64::from(bits);
            }
            RowPlaneContent::Groups { .. } => {
                panic!(
                    "row plane `{}` holds no column's codes",
                    self.planes[plane].name
                )
            }
        };
        let bits = u64::from(bits);
        let stride = geometry.stride;
        let offset = geometry.offsets[plane];
        match self.layout {
            Layout::Rows8 if !grids => {
                let group = u64::from(self.packet.group);
                let (chunk, within) = (32 / bits, column % group);
                let byte = self.span_byte(geometry, row, column / group, plane)
                    + within / chunk * 32
                    + row % 8 * 4
                    + within % 4;
                byte * 8 + within % chunk / 4 * bits
            }
            Layout::Rows16 | Layout::Rows8 => {
                let group = u64::from(self.packet.group);
                let byte = self
                    .share_byte(geometry, row, column / group, plane)
                    .expect("a rows16 share, or a rows8 grid share, is its bytes");
                byte * 8 + column % group / unit * bits
            }
            Layout::Rows32 => interleaved(
                ROWS32_TILE_ROWS,
                ROWS32_CHUNK / unit,
                stride,
                offset,
                row,
                column / unit,
                bits,
            ),
            Layout::Mma16 => {
                let row_bytes = geometry.bytes_per_row[plane];
                let (tile, r) = (row / MMA_TILE_ROWS, row % MMA_TILE_ROWS);
                let (block, c) = (column / MMA_KBLOCK, column % MMA_KBLOCK);
                let bit = if grids {
                    ((block * MMA_TILE_ROWS + r) * (MMA_KBLOCK / unit) + c / unit) * bits
                } else {
                    let (step, k) = (c / MMA_STEP, c % MMA_STEP);
                    let lane = 4 * (r % 8) + (k % 8) / 2;
                    let slot = (k / 8) * 2 + r / 8 + 4 * (k % 2);
                    Self::mma_fragment_byte(block, lane, bits, step) * 8 + bits * slot
                };
                let (byte, within) = (bit / 8, bit % 8);
                let stored = tile * MMA_TILE_ROWS + byte / row_bytes;
                (stored * stride + offset + byte % row_bytes) * 8 + within
            }
            Layout::Packet => unreachable!("a row layout is never `packet`"),
        }
    }

    /// The first byte of storage group `packet`'s span of row plane `plane`
    /// in the tile of stored row `row`, in the layouts whose tile keeps a
    /// group's shares of its rows together (`rows16`, a row its tile, and
    /// `rows8`): the spans of a tile's plane follow one another, group by
    /// group.
    pub fn span_byte(&self, geometry: &RowGeometry, row: u64, packet: u64, plane: usize) -> u64 {
        let tile = self.tile_rows();
        assert!(
            matches!(self.layout, Layout::Rows16 | Layout::Rows8),
            "`{}` keeps no group span",
            self.layout.as_str()
        );
        row / tile * geometry.stride * tile
            + geometry.offsets[plane] * tile
            + packet * u64::from(self.planes[plane].bytes_per_group) * tile
    }

    /// The first byte of storage group `packet`'s share of row plane
    /// `plane` of stored row `row` (its `bytes_per_group` bytes, one after
    /// another), in the layouts that keep a share's bytes together: every
    /// plane of `rows16`, a grid plane of `rows8` (whose span holds its
    /// rows' shares in turn), and the group planes of `rows32` and `mma16`.
    /// `None` for the code and group planes of `rows8` and a code plane of
    /// `rows32` or `mma16` (`code_bit`, `packet_bit`).
    pub fn share_byte(
        &self,
        geometry: &RowGeometry,
        row: u64,
        packet: u64,
        plane: usize,
    ) -> Option<u64> {
        let group_bytes = u64::from(self.planes[plane].bytes_per_group);
        let (stride, offset, tile) = (geometry.stride, geometry.offsets[plane], self.tile_rows());
        let groups = matches!(self.planes[plane].content, RowPlaneContent::Groups { .. });
        Some(match self.layout {
            Layout::Rows16 => self.span_byte(geometry, row, packet, plane),
            Layout::Rows8 if matches!(
                self.planes[plane].content,
                RowPlaneContent::Grids { .. } | RowPlaneContent::GridHigh { .. }
            ) => self.span_byte(geometry, row, packet, plane) + row % tile * group_bytes,
            Layout::Rows8 => return None,
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

    placement_rule! {
        /// The byte of entry `entry` (of eight) of k-block `block` (of four) of
        /// row `r` of its tile within a storage group's entries of a `records`
        /// tile (`record_byte`'s GEMV lane order): half `block / 2`, lane
        /// `4 (r % 8) + entry % 4`, word `2 (block % 2) + r / 8`, its low or
        /// high half by `entry / 4`.
        record_entry, RECORD_ENTRY_RULE(r, block, entry) =
            (block) / 2 * 512 + (4 * ((r) % 8) + (entry) % 4) * 16 + ((block) % 2 * 2 + (r) / 8) * 4 + (entry) / 4 * 2
    }

    placement_rule! {
        /// The place of row `r` of its tile among a storage group's factors of
        /// a `records` tile (`[g][row g, row g + 8]`), in factor widths.
        record_factor, RECORD_FACTOR_RULE(r) = 2 * ((r) % 8) + (r) / 8
    }

    placement_rule! {
        /// The row of a tile that slot `slot` of lane `lane`'s `k16` step of
        /// an `mma16` code plane holds (`code_bit`, inverted).
        mma_fragment_row, MMA_FRAGMENT_ROW_RULE(lane, slot) = (lane) / 4 + 8 * ((slot) % 2)
    }

    placement_rule! {
        /// The column of a `k16` step that slot `slot` of lane `lane`'s step
        /// of an `mma16` code plane holds (`code_bit`, inverted).
        mma_fragment_column, MMA_FRAGMENT_COLUMN_RULE(lane, slot) =
            2 * ((lane) % 4) + 8 * (((slot) % 4) / 2) + (slot) / 4
    }

    placement_rule! {
        /// The byte of `k16` step `step` of lane `lane` of k-block `block` in
        /// an `mma16` code plane's tile sequence, `bits` a column (`code_bit`).
        mma_fragment_byte, MMA_FRAGMENT_BYTE_RULE(block, lane, bits, step) =
            ((block) * 32 + (lane)) * 4 * (bits) + (bits) * (step)
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
    /// order. A vector-code entry's bits past its whole bytes are in the grid
    /// high plane, an entry's in entry order.
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
            let (grid, carried, factors) = match &self.planes[index].content {
                RowPlaneContent::Grids { grid, carried, .. } => (*grid, *carried, None),
                RowPlaneContent::Records {
                    grid,
                    carried,
                    factors,
                    ..
                } => (*grid, Some(*carried), Some(factors)),
                _ => unreachable!(),
            };
            let bits = self.column_bits(index);
            if carried == Some(plane as u32) {
                let codes = &self.packet.planes[plane];
                let (field, j) = (bit / codes.entry_bits, bit % codes.entry_bits);
                let entry = field * (codes.group / GRID_VALUES) + j;
                return self.entry_bit(index, entry, grid.carried_bit());
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

    /// The row plane and share bit of bit `bit` of vector-code entry `entry`
    /// of a storage group, whose whole bytes are in entries plane `entries`
    /// (`group_bit`) and the rest in the grid high plane.
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

    /// Bits of a storage group's codes code plane `plane` holds: its
    /// columns' bits, or a grid high plane's entries' (a records plane's
    /// factors are not codes).
    pub fn code_bits(&self, plane: usize) -> u32 {
        match self.planes[plane].content {
            RowPlaneContent::Codes { .. }
            | RowPlaneContent::Grids { .. }
            | RowPlaneContent::Records { .. } => self.column_bits(plane) * self.packet.group,
            RowPlaneContent::GridHigh { bits, .. } => bits * self.packet.group / GRID_VALUES,
            RowPlaneContent::Groups { .. } => {
                panic!("row plane `{}` holds no codes", self.planes[plane].name)
            }
        }
    }

    /// Bits a column of code plane `plane` holds.
    pub(crate) fn column_bits(&self, plane: usize) -> u32 {
        match self.planes[plane].content {
            RowPlaneContent::Codes { bits, .. }
            | RowPlaneContent::Grids { bits, .. }
            | RowPlaneContent::Records { bits, .. } => bits,
            RowPlaneContent::GridHigh { .. } | RowPlaneContent::Groups { .. } => {
                panic!(
                    "row plane `{}` holds no column's codes",
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
            // The share bit's entry, placed at its first column.
            RowPlaneContent::GridHigh { bits, .. } => {
                let column =
                    packet * u64::from(self.packet.group) + u64::from(share / bits * GRID_VALUES);
                self.code_bit(geometry, index, row, column) + u64::from(share % bits)
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
                    Self::record_factor(r) * width + u64::from(within / 8)
                });
                byte * 8 + u64::from(within % 8)
            }
            RowPlaneContent::Groups { planes, .. } if self.layout == Layout::Rows8 => {
                self.span_byte(geometry, row, packet, index) * 8 + self.record_bit(planes, row % 8, share)
            }
            RowPlaneContent::Groups { .. } => {
                let byte = self
                    .share_byte(geometry, row, packet, index)
                    .expect("a group plane's share is its bytes");
                byte * 8 + u64::from(share)
            }
        }
    }

    /// The bit of a `rows8` group plane's span (its `planes`' shares of a
    /// tile's eight rows) holding share bit `share` of row `r`: each listed
    /// plane's records (a coefficient group's fields, `fields *
    /// entry_bits`), record by record, each of the eight rows in turn, then
    /// the rows' bits past whole records.
    fn record_bit(&self, planes: &[u32], r: u64, share: u32) -> u64 {
        let mut prefix = 0u64;
        for member in planes {
            let plane = &self.packet.planes[*member as usize];
            let bits = u64::from(plane.bytes_per_group) * 8;
            let within = u64::from(share) - prefix;
            if within < bits {
                let width = u64::from(plane.fields * plane.entry_bits);
                let records = u64::from(self.packet.group / plane.group);
                let (record, bit) = (within / width, within % width);
                let tail = bits - records * width;
                let placed = if record < records {
                    (record * 8 + r) * width + bit
                } else {
                    records * width * 8 + r * tail + bit
                };
                return prefix * 8 + placed;
            }
            prefix += bits;
        }
        unreachable!("a group plane's share bit lies in one of its planes")
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

    /// Whether the layout stores its tiles as records (its only plane a
    /// `RowPlaneContent::Records`).
    pub fn records(&self) -> bool {
        self.planes.iter().any(|plane| matches!(plane.content, RowPlaneContent::Records { .. }))
    }

    /// The packet plane a carried grid layout's entries carry, and the entry
    /// bit carrying it (`GridCodes::carried_bit`).
    pub fn carried(&self) -> Option<(u32, u32)> {
        self.planes.iter().find_map(|candidate| match candidate.content {
            RowPlaneContent::Grids {
                grid,
                carried: Some(plane),
                ..
            }
            | RowPlaneContent::Records {
                grid,
                carried: plane,
                ..
            } => Some((plane, grid.carried_bit())),
            _ => None,
        })
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
    bytes.checked_next_multiple_of(ROW_ALIGNMENT)
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

/// The stores storage of one representation takes: the one write rule every
/// checker and lowering site asks.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StoreRule {
    /// Writable storage (`RepresentationAccess::ReadWrite`) takes every store.
    Any,
    /// Packed storage (`Packed`, `PackedRows`) has no canonical logical
    /// write: it takes exactly the store of whole trailing matrices (its last
    /// two axes) from a value of this same representation, whose encoded
    /// bytes the store copies.
    Records(RepresentationId),
    /// Every other storage of this representation takes no store.
    Refused(RepresentationId),
}

impl StoreRule {
    /// A store selecting whole trailing matrices (`whole_records`), or any
    /// other part: taken, or refused with the diagnostic of its rule.
    pub fn admits(self, whole_records: bool) -> Result<(), String> {
        match self {
            Self::Any => Ok(()),
            Self::Records(_) if whole_records => Ok(()),
            Self::Records(own) => Err(decode_only(own)),
            Self::Refused(own) => Err(storeless(own)),
        }
    }

    /// A stored value of representation `value` (`None` for a value that is
    /// not a tensor of one): taken, or refused with the diagnostic of its
    /// rule.
    pub fn admits_value(self, value: Option<RepresentationId>) -> Result<(), String> {
        match self {
            Self::Any => Ok(()),
            Self::Records(own) if value == Some(own) => Ok(()),
            Self::Records(own) => Err(decode_only(own)),
            Self::Refused(own) => Err(storeless(own)),
        }
    }
}

/// The diagnostic of a store packed storage of `id` does not take.
fn decode_only(id: RepresentationId) -> String {
    format!(
        "representation `{}` is decode-only: it is written only by whole trailing matrices of the same representation",
        representation_info(id).name
    )
}

/// The diagnostic of any store into storage of `id`, which takes none.
fn storeless(id: RepresentationId) -> String {
    format!("representation `{}` is read-only: it takes no store", representation_info(id).name)
}

pub fn store_rule(id: RepresentationId) -> StoreRule {
    let info = representation_info(id);
    match info.kind {
        _ if info.access == RepresentationAccess::ReadWrite => StoreRule::Any,
        RepresentationKind::Packed(_) | RepresentationKind::PackedRows(_) => {
            StoreRule::Records(id)
        }
        _ => StoreRule::Refused(id),
    }
}

/// Whether storage of `id` takes some store, so binds a `&mut` parameter.
pub fn record_writable(id: RepresentationId) -> bool {
    store_rule(id).admits(true).is_ok()
}

pub fn representations() -> &'static [RepresentationInfo] {
    internals::representations()
}
/// The destination dtype of `repack` between two dense float representations:
/// an element conversion, each element decoded to F32 and rounded into the
/// destination (as `import_dense` imports a dense weight). `None` unless both
/// are dense floats, whose exact conversions are the registered ones
/// (`representation_conversion`).
pub fn dense_conversion(source: RepresentationId, destination: RepresentationId) -> Option<DType> {
    let float = |id| match representation_info(id).kind {
        RepresentationKind::Dense(dtype) if dtype.is_float() => Some(dtype),
        _ => None,
    };
    float(source).and(float(destination))
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
/// Where the scales of a GGUF source's values are: in its stored tensor's
/// own blocks, or in a companion grid of block scales.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum GgufScales {
    /// In the tensor's blocks (`magnitude_ggml` geometry), or none.
    Own,
    /// A companion grid of scales, one per tile of the source's values,
    /// over a tensor of single-value blocks: stored planar, each matrix's
    /// codes then its grid, as the file stores the two tensors
    /// ([`GridPlanes`]).
    Grid(Grid),
}

impl GgufScales {
    /// Every place scales can be: a tensor's own blocks, then each
    /// registered grid.
    pub const ALL: [Self; 4] = [
        Self::Own,
        Self::Grid(Grid::REGISTERED[0]),
        Self::Grid(Grid::REGISTERED[1]),
        Self::Grid(Grid::REGISTERED[2]),
    ];
}

/// A companion grid: the element of its scales and the `[rows, columns]`
/// tile of values each one scales.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Grid {
    pub scale: GridScale,
    pub tile: [u64; 2],
}

impl Grid {
    /// The grids real files carry, each a source of its own: UE8M0 scales
    /// per row of 32 (DeepSeek V4.1's n-gram table) and per 32×32 tile
    /// (its other FP8 weights), and F32 scales per 128×128 tile (an FP8
    /// checkpoint's `weight_scale_inv`). A tile is registered when a file
    /// needs it; a weight under any other is refused when planned.
    pub const REGISTERED: [Self; 3] = [
        Self { scale: GridScale::E8m0, tile: [1, 32] },
        Self { scale: GridScale::E8m0, tile: [32, 32] },
        Self { scale: GridScale::F32, tile: [128, 128] },
    ];
}

/// The scale element of a companion grid (`GgufScales::Grid`), the first
/// bytes of a source packet (the scale, then the codes it scales).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum GridScale {
    /// One E8M0 byte, `2^(e - 127)`, 0xff NaN (OCP MX).
    E8m0,
    /// One F32 value, little-endian.
    F32,
}

impl GridScale {
    /// The bytes of one scale.
    pub const fn bytes(self) -> u64 {
        match self {
            Self::E8m0 => 1,
            Self::F32 => 4,
        }
    }

    /// Its short name, as a source's name carries it
    /// (`gguf_f8_e4m3_e8m0_32x32`).
    pub const fn name(self) -> &'static str {
        match self {
            Self::E8m0 => "e8m0",
            Self::F32 => "f32",
        }
    }
}

/// A GGUF source the registry converts from: its ggml type, where its
/// scales are, its storage name, and the resident representation its
/// conversion writes. A source packet is one block of the type
/// (`magnitude_ggml` geometry), or under a grid (`GgufScales::Grid`) a
/// scale and the codes it scales, which lie in two planes
/// ([`GridPlanes`]).
#[derive(Clone, Copy, Debug)]
struct GgufSource {
    encoding: Encoding,
    scales: GgufScales,
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
        scales: GgufScales::Own,
        name,
        resident,
    }
}

const GGUF_SOURCES: [GgufSource; 28] = [
    gguf_source_spec(Encoding::Q3K, "gguf_q3_k", "q3k"),
    gguf_source_spec(Encoding::Iq3S, "gguf_iq3_s", "iq3s"),
    gguf_source_spec(Encoding::Q4K, "gguf_q4_k", "q4k"),
    gguf_source_spec(Encoding::Q5K, "gguf_q5_k", "q5k"),
    gguf_source_spec(Encoding::Q6K, "gguf_q6_k", "q6k"),
    gguf_source_spec(Encoding::Q8_0, "gguf_q8_0", "q8g32s"),
    gguf_source_spec(Encoding::Iq4Nl, "gguf_iq4_nl", "iq4g32s"),
    gguf_source_spec(Encoding::Iq4Xs, "gguf_iq4_xs", "iq4xs"),
    gguf_source_spec(Encoding::Q4_0, "gguf_q4_0", "q4g32s"),
    gguf_source_spec(Encoding::Q5_0, "gguf_q5_0", "q5g32s"),
    gguf_source_spec(Encoding::Q5_1, "gguf_q5_1", "q5g32"),
    gguf_source_spec(Encoding::Mxfp4, "gguf_mxfp4", "mxfp4g32"),
    gguf_source_spec(Encoding::Nvfp4, "gguf_nvfp4", "nvfp4g16"),
    gguf_source_spec(Encoding::Iq2Xs, "gguf_iq2_xs", "iq2xs"),
    gguf_source_spec(Encoding::Iq3Xxs, "gguf_iq3_xxs", "iq3xxs"),
    gguf_source_spec(Encoding::Iq2Xxs, "gguf_iq2_xxs", "iq2xxs"),
    gguf_source_spec(Encoding::Q4_1, "gguf_q4_1", "q4g32"),
    gguf_source_spec(Encoding::Q2K, "gguf_q2_k", "q2k"),
    gguf_source_spec(Encoding::Tq2_0, "gguf_tq2_0", "q2g256"),
    gguf_source_spec(Encoding::Q1_0, "gguf_q1_0", "q1g128"),
    gguf_source_spec(Encoding::Q2_0, "gguf_q2_0", "q2g64"),
    gguf_source_spec(Encoding::Iq2S, "gguf_iq2_s", "iq2s"),
    gguf_source_spec(Encoding::Iq1S, "gguf_iq1_s", "iq1s"),
    gguf_source_spec(Encoding::Iq1M, "gguf_iq1_m", "iq1m"),
    // Base-3 digits reside as the 2-bit codes of their radix: no
    // representation stores digits, since a value's digit costs a product
    // more to decode than a bit field, and no backend's product is bound by
    // the 18% of bytes they save (spec 26-10-30 §13).
    gguf_source_spec(Encoding::Tq1_0, "gguf_tq1_0", "q2g256"),
    // E4M3 codes under their companion grid (a packet: the scale, then a
    // group of codes), moved bit for bit: under E8M0 scales OCP MXFP8's
    // block; under F32 scales (an FP8 checkpoint's `weight_scale_inv`) a
    // stored F32 scale per 128, every scale of a 128-column tile. A
    // source's name states its grid's scale and tile.
    GgufSource {
        encoding: Encoding::F8E4M3,
        scales: GgufScales::Grid(Grid::REGISTERED[0]),
        name: "gguf_f8_e4m3_e8m0_1x32",
        resident: "mxfp8g32",
    },
    GgufSource {
        encoding: Encoding::F8E4M3,
        scales: GgufScales::Grid(Grid::REGISTERED[1]),
        name: "gguf_f8_e4m3_e8m0_32x32",
        resident: "mxfp8g32",
    },
    GgufSource {
        encoding: Encoding::F8E4M3,
        scales: GgufScales::Grid(Grid::REGISTERED[2]),
        name: "gguf_f8_e4m3_f32_128x128",
        resident: "fp8g128",
    },
];

impl GgufSource {
    /// The bytes and the values of one source packet: a block of the type,
    /// or under a grid its scale code then the codes of one group of the
    /// resident representation (the values a scale of it scales), which
    /// lie in two planes ([`GridPlanes`]).
    fn packet(&self) -> (u64, u64) {
        match self.scales {
            GgufScales::Own => (self.encoding.block_bytes(), self.encoding.block_elements()),
            GgufScales::Grid(Grid { scale, .. }) => {
                assert_eq!(self.encoding.block_elements(), 1, "a scale grid scales single values");
                let group = crate::repr::REPRS
                    .iter()
                    .find(|repr| repr.name == self.resident)
                    .expect("a GGUF source names a registered resident representation")
                    .group;
                let group = u64::from(group);
                (scale.bytes() + group * self.encoding.block_bytes(), group)
            }
        }
    }
}

/// The registered source of `encoding` stored with its scales where
/// `scales` says, whatever its rows' width; `None` when the registry
/// converts from none.
pub fn gguf_packet_source(encoding: Encoding, scales: GgufScales) -> Option<RepresentationId> {
    GGUF_SOURCES
        .iter()
        .find(|source| source.encoding == encoding && source.scales == scales)
        .and_then(|source| representation(source.name))
}

/// The external storage rows of `width` stored values of `encoding`,
/// scaled as `scales` says, are read as: the registered source of the
/// pair ([`gguf_packet_source`]) when its packet divides the row (a GGUF
/// row is whole blocks).
pub fn gguf_source(encoding: Encoding, scales: GgufScales, width: u64) -> Option<RepresentationId> {
    gguf_packet_source(encoding, scales).filter(|&source| {
        let packet = external_packet(source).expect("a GGUF source is external storage");
        width.is_multiple_of(u64::from(packet.logical_group))
    })
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
impl Table {
    /// The key of the entry of this table that `name` names among its rows
    /// (a representation's name; a conversion's endpoints; a capability's
    /// backend and namespace; an intrinsic's capability, name and overload):
    /// the low 31 bits of a SHA-256 of the table and the name, so that a
    /// native source carries it as a signed 32-bit literal. A function of
    /// the entry's own name alone, it is the same in every registry that
    /// holds the entry, whatever else it holds. Every key of a table is
    /// distinct and nonzero (`internals::Positions`).
    pub(crate) fn key(self, name: &str) -> u32 {
        use sha2::Digest as _;
        let mut hash = sha2::Sha256::new();
        hash.update(b"seismic-registry-key-v1\0");
        hash.update(match self {
            Self::Capability => "capability",
            Self::Intrinsic => "intrinsic",
            Self::Representation => "representation",
            Self::RepresentationConversion => "representation-conversion",
        });
        hash.update([0]);
        hash.update(name.as_bytes());
        let digest: [u8; 32] = hash.finalize().into();
        u32::from_le_bytes([digest[0], digest[1], digest[2], digest[3]]) & 0x7fff_ffff
    }
}
/// The position in this process's `table` of its entry keyed `key`; `None`
/// when no entry of it is.
pub(crate) fn position(table: Table, key: u32) -> Option<u32> {
    internals::position(table, key)
}
/// The registry's own `'static` copy of an intrinsic or opaque-value name it
/// declares, or `None` when the registry declares no such name.
pub(crate) fn declared_name(name: &str) -> Option<&'static str> {
    internals::declared_name(name)
}
/// A name the registry declares. Fields of this type serialize through
/// [`serialize_declared_name`] and deserialize through
/// [`deserialize_declared_name`]; naming the type keeps serde from treating
/// the field as borrowed from the input.
pub(crate) type DeclaredName = &'static str;
/// Serializes a registry-declared name as the string it is, exactly as the
/// derive writes a `&str`.
pub(crate) fn serialize_declared_name<S: serde::Serializer>(
    name: &DeclaredName,
    serializer: S,
) -> Result<S::Ok, S::Error> {
    serializer.serialize_str(name)
}
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
/// The storage units of a contiguous tensor with `extents` in
/// representation `id`, per axis, and the bytes of one unit: dense units
/// are elements; a packed or external tensor's packing axis counts packets
/// (`⌈last / group⌉` of the packet's bytes); a row layout's leading extents
/// count rows (the row axis padded to whole tiles) and its packing axis is
/// one unit of the row stride (`PackedRowLayout::storage_units`,
/// `row_stride_bytes`). `None` where `extents` is empty for a packed or
/// external representation, below a row layout's rank, or on overflow.
pub fn storage_units(id: RepresentationId, extents: &[u64]) -> Option<(Vec<u64>, u64)> {
    let packets = |group: u32, packet_size: u32| {
        let mut units = extents.to_vec();
        let last = units.last_mut()?;
        *last = last.div_ceil(u64::from(group));
        Some((units, u64::from(packet_size)))
    };
    match &representation_info(id).kind {
        RepresentationKind::Dense(dtype) => Some((extents.to_vec(), u64::from(dtype.bytes()))),
        RepresentationKind::Packed(layout) => packets(layout.group, layout.packet_size),
        // A source under a companion grid is stored planar: a matrix's codes
        // then its grid are one unit, so the last two axes count that one
        // unit and the leading axes count matrices. An empty matrix is no
        // unit: a zero extent, as of any tensor that holds nothing.
        RepresentationKind::External(layout) if layout.planes.is_some() => {
            let matrices = layout.matrices(extents)?;
            let mut units = extents[..extents.len() - 2].to_vec();
            if matrices.bytes == 0 {
                units.extend([0, 1]);
                Some((units, 1))
            } else {
                units.extend([1, 1]);
                Some((units, matrices.bytes))
            }
        }
        RepresentationKind::External(layout) => packets(layout.logical_group, layout.packet_size),
        RepresentationKind::PackedRows(layout) => {
            Some((layout.storage_units(extents)?, layout.row_stride_bytes(*extents.last()?)?))
        }
    }
}

/// The bytes of contiguous storage `units` of `width` bytes each, `None`
/// where a row-major stride over them overflows: their product accumulated
/// from the last axis, as their strides are, times the width.
pub fn units_bytes(units: &[u64], width: u64) -> Option<u64> {
    units
        .iter()
        .rev()
        .try_fold(1u64, |count, extent| count.checked_mul(*extent))?
        .checked_mul(width)
}

/// Canonical storage bytes of a contiguous tensor with `extents` in
/// representation `id`: its storage units' bytes ([`storage_units`],
/// [`units_bytes`]). `None` where it has no storage units or a row-major
/// stride over them overflows `u64`.
pub fn canonical_bytes(id: RepresentationId, extents: &[u64]) -> Option<u64> {
    let (units, width) = storage_units(id, extents)?;
    units_bytes(&units, width)
}

/// Bytes of one record (a row of the leading axis) of a tensor of
/// representation `id` with `extents`, where every record lies in its own
/// consecutive bytes, one after another ([`canonical_bytes`] of the trailing
/// extents), whatever its count of records, none included (a view of no
/// records binds a record port as a table of no entries). `None` for a
/// tensor of empty records, and for one whose layout interleaves its
/// leading axis: a packed or external tensor's only axis (its packing
/// axis), or a row layout's tiled rows.
pub fn record_bytes(id: RepresentationId, extents: &[u64]) -> Option<std::num::NonZeroU64> {
    let (_, trailing) = extents.split_first()?;
    let own = match &representation_info(id).kind {
        RepresentationKind::Dense(_) => true,
        RepresentationKind::Packed(_) => !trailing.is_empty(),
        // A planar source's rows share its matrix's grid, so its records are
        // matrices.
        RepresentationKind::External(layout) => match layout.planes {
            None => !trailing.is_empty(),
            Some(_) => trailing.len() >= 2,
        },
        RepresentationKind::PackedRows(layout) => {
            trailing.len() >= 2 || (trailing.len() == 1 && layout.tile_rows() == 1)
        }
    };
    if !own {
        return None;
    }
    std::num::NonZeroU64::new(canonical_bytes(id, trailing)?)
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

/// The packet of external (artifact-format) storage `id`; `None` for any
/// other storage.
pub fn external_packet(id: RepresentationId) -> Option<&'static ExternalPacketLayout> {
    match &representation_info(id).kind {
        RepresentationKind::External(packet) => Some(packet),
        _ => None,
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
    //! `intrinsics::capability_rows` and `repr::REPRS` tables; every id is
    //! its entry's key (`Table::key`) and its position in these vectors, and
    //! is valid for the process lifetime.
    //! Capabilities are contiguous per backend and intrinsics contiguous per
    //! capability so backend/capability slices are direct sub-slices.

    use super::*;
    use crate::external_tables::TQ1_DIGITS;
    use crate::intrinsics::{capability_rows, RowOperand, RowResult};
    use crate::repr::{Coding, PlaneEncoding, Repr, REPRS};
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
        conversions: OnceLock<(Vec<RepresentationConversion>, Positions)>,
        capability_positions: Positions,
        intrinsic_positions: Positions,
        representation_positions: Positions,
    }

    /// The position of each entry of one table by its key, built with the
    /// table: two entries of one key, or a zero key (a native source's
    /// absent tensor), are an invariant violation of the registry.
    pub(super) struct Positions(std::collections::HashMap<u32, u32>);

    impl Positions {
        fn of(table: Table, keys: impl IntoIterator<Item = (u32, usize)>) -> Self {
            let mut positions = std::collections::HashMap::new();
            for (key, position) in keys {
                assert_ne!(key, 0, "a {table:?} entry is keyed zero");
                let position = u32::try_from(position).expect("a registry table fits u32 positions");
                if let Some(other) = positions.insert(key, position) {
                    panic!("{table:?} entries {other} and {position} share key {key:08x}: rename one");
                }
            }
            Self(positions)
        }
    }

    /// The id of the entry named `name` (`Table::key`) at `position` of
    /// `table`.
    fn identify<Id>(table: Table, name: &str, position: usize, id: fn(u32, u32) -> Id) -> Id {
        id(
            table.key(name),
            u32::try_from(position).expect("a registry table fits u32 positions"),
        )
    }

    fn representation_id(name: &str, position: usize) -> RepresentationId {
        identify(Table::Representation, name, position, RepresentationId::new)
    }

    fn build() -> Tables {
        // Representations: every dense dtype first (ordinal = dtype ordinal),
        // then the packed representations in `REPRS` order.
        let mut representations = Vec::new();
        for dtype in DType::ALL {
            representations.push(RepresentationInfo {
                id: representation_id(dtype.name(), representations.len()),
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
                let name = element_name(repr.name, operands, Layout::Packet);
                representations.push(RepresentationInfo {
                    id: representation_id(name, representations.len()),
                    name,
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
        // representation). Every source converts into a representation of
        // its own block, moving its codes and coefficient fields bit for bit
        // (IQ2_XS's and IQ3_XXS's factors widened exactly, TQ1_0's digits
        // looked up as 2-bit codes).
        let external_specs = GGUF_SOURCES.map(|source| {
            let (bytes, values) = source.packet();
            let packet_size = u32::try_from(bytes).expect("a GGUF source packet fits u32 bytes");
            let logical_group =
                u32::try_from(values).expect("a GGUF source packet fits u32 values");
            let planes = match source.scales {
                GgufScales::Own => None,
                GgufScales::Grid(Grid { scale, tile }) => {
                    let tile = tile.map(|extent| u32::try_from(extent).expect("a grid tile fits u32"));
                    assert!(tile[1].is_multiple_of(logical_group), "a grid tile is whole packets of a row");
                    // Every window groups a tile's rows by 16-row tiles.
                    assert!(tile[0] == 1 || tile[0].is_multiple_of(16), "a grid tile's rows are one or whole 16-row tiles");
                    Some(GridPlanes {
                        scale_bytes: u32::try_from(scale.bytes()).expect("a scale fits u32 bytes"),
                        tile,
                    })
                }
            };
            (source.name, packet_size, logical_group, source.resident, planes)
        });
        // A GGUF packet is declared aligned to the largest power of two, to
        // 4 (the widest unit a kernel reads a source packet in), dividing
        // its size. Device allocations are aligned further; a tensor's data
        // in its file starts at a multiple of `general.alignment`, any power
        // of two, so a view of the file's pages that falls short is not
        // made, and the importer uploads the tensor instead (the engine's
        // `ImportSource::acquire`).
        for (name, packet_size, logical_group, _, planes) in external_specs {
            representations.push(RepresentationInfo {
                id: representation_id(name, representations.len()),
                name,
                representation: name,
                layout: Layout::Packet,
                kind: RepresentationKind::External(ExternalPacketLayout {
                    packet_size,
                    packet_alignment: crate::repack::alignment(packet_size).min(4),
                    logical_group,
                    packing_axis: PackingAxis::Last,
                    planes,
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
        for (_, _, _, resident, _) in external_specs {
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
                    let name = element_name(resident, operands, layout);
                    representations.push(RepresentationInfo {
                        id: representation_id(name, representations.len()),
                        name,
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
        let representation_positions = Positions::of(
            Table::Representation,
            representations.iter().map(|info| (info.id.key(), info.id.position())),
        );
        let dense_id = |dtype: DType| representations[usize::from(dtype.ordinal())].id;

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
                        let id = identify(
                            Table::Capability,
                            &format!("{}/{}", backend.as_str(), row.capability),
                            capabilities.len(),
                            CapabilityId::new,
                        );
                        capabilities.push(CapabilityInfo {
                            id,
                            backend,
                            name: row.capability,
                        });
                        intrinsic_ranges.push((intrinsics.len(), intrinsics.len()));
                        id
                    }
                };
                // An overload is named by its place among the earlier rows
                // of its capability and name.
                let overload = overloads
                    .iter()
                    .find(|(owner, name, _)| *owner == capability && *name == row.name)
                    .map_or(0, |(_, _, rows)| rows.len());
                let id = identify(
                    Table::Intrinsic,
                    &format!("{}/{}/{}#{overload}", backend.as_str(), row.capability, row.name),
                    intrinsics.len(),
                    IntrinsicId::new,
                );
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
                intrinsic_ranges[capability.position()].1 = intrinsics.len();
                match overloads
                    .iter_mut()
                    .find(|(owner, name, _)| *owner == capability && *name == row.name)
                {
                    Some((_, _, rows)) => {
                        let categories = |row: IntrinsicId| {
                            intrinsics[row.position()]
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
        let capability_positions = Positions::of(
            Table::Capability,
            capabilities.iter().map(|info| (info.id.key(), info.id.position())),
        );
        let intrinsic_positions = Positions::of(
            Table::Intrinsic,
            intrinsics.iter().map(|signature| (signature.id.key(), signature.id.position())),
        );
        Tables {
            capabilities,
            capability_ranges,
            intrinsics,
            denotations,
            intrinsic_ranges,
            overloads,
            representations,
            conversions: OnceLock::new(),
            capability_positions,
            intrinsic_positions,
            representation_positions,
        }
    }

    /// The registered conversions between `representations`: of each
    /// external source into every element of its resident representation.
    fn conversions(
        representations: &'static [RepresentationInfo],
    ) -> (Vec<RepresentationConversion>, Positions) {
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
                let destination_layout = ConversionLayout::of(&destination.kind)
                    .expect("external conversion destination is packed");
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
                    id: identify(
                        Table::RepresentationConversion,
                        &format!("{}->{}", source.name, destination.name),
                        conversions.len(),
                        RepresentationConversionId::new,
                    ),
                    source: source.id,
                    destination: destination.id,
                    source_layout,
                    destination_layout,
                    recipe,
                });
            }
        }
        let positions = Positions::of(
            Table::RepresentationConversion,
            conversions.iter().map(|conversion| (conversion.id.key(), conversion.id.position())),
        );
        (conversions, positions)
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
                PlaneEncoding::Packed { .. }
                | PlaneEncoding::Grid(_)
                | PlaneEncoding::SignedLocal { .. } => 4,
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
    ///    their whole bytes, and `grid_high`, the rest of each entry, carrying
    ///    the `coefficients` plane when the representation's coefficients are
    ///    carried;
    /// 2. `scales`: the hierarchical local coefficients exactly as the packet
    ///    `coefficients` plane packs them (never widened), unless carried;
    /// 3. `supers`: every remaining coefficient plane of one packet
    ///    concatenated per group (k-quant super factors `d, dmin`; a direct
    ///    per-group scale).
    ///
    /// `mma16` of a carried representation of whole-byte entries is one
    /// `records` plane holding the entries, their carried codes and the
    /// supers.
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
        let carried = match &repr.coding {
            Coding::Hierarchical {
                locals: LocalCodes::Carried { grid, .. },
                ..
            } => {
                let [plane] = scales[..] else {
                    panic!("`{}` carries one coefficient plane", repr.name)
                };
                let codes = &packet.planes[plane as usize];
                assert!(
                    codes.entry_bits == codes.group / GRID_VALUES
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
            // A carried grid representation's `mma16` tile of whole-byte
            // entries is records.
            (PlaneEncoding::Grid(grid), bits)
                if layout == Layout::Mma16 && carried.is_some() && bits % 8 == 0 =>
            {
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
                        grid,
                        carried: carried.unwrap(),
                        factors: supers.clone(),
                    },
                }]
            }
            // Vector codes stay whole entries, one entry several values'
            // codes: their whole bytes, then the rest.
            (PlaneEncoding::Grid(grid), bits) => {
                let (low, high) = (bits / 8 * 8, bits % 8);
                let entries = group / GRID_VALUES;
                assert!(
                    low > 0 && (entries * high).is_multiple_of(8),
                    "`{}` grid entries hold whole bytes, and their rest whole bytes a group",
                    repr.name
                );
                let mut planes = vec![RowPlaneInfo {
                    name: "grids",
                    bytes_per_group: entries * low / 8,
                    content: RowPlaneContent::Grids {
                        bits: low / GRID_VALUES,
                        grid,
                        carried,
                    },
                }];
                if high > 0 {
                    planes.push(RowPlaneInfo {
                        name: "grid_high",
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
                | RowPlaneContent::Grids { carried: None, .. }
                | RowPlaneContent::GridHigh { .. } => Vec::new(),
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
        if layout == Layout::Rows8 {
            for plane in &planes {
                if let RowPlaneContent::Codes { bits, .. } = plane.content {
                    assert!(
                        matches!(bits, 1 | 2 | 4 | 8) && plane.bytes_per_group % 4 == 0,
                        "`{}` rows8 code planes hold whole panel units",
                        repr.name
                    );
                }
            }
        }
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
                            RowPlaneContent::Codes { .. } => {
                                multiple / gcd(multiple, chunk) * chunk
                            }
                            // A tile's k-block of a grids plane is its 16
                            // rows' k-block bytes, whole chunks; records
                            // run over the tile's rows whole.
                            RowPlaneContent::Grids { .. }
                            | RowPlaneContent::GridHigh { .. }
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

    fn direct_bits(byte: u32, bytes: u32) -> Vec<Option<u32>> {
        (0..bytes * 8).map(|bit| Some(byte * 8 + bit)).collect()
    }

    /// The bit routes of ggml's 2-bit codes of 256 values from byte `byte`:
    /// value p's code is at bit 2 ((p % 128) / 32) of byte 32 (p / 128) +
    /// p % 32 of them (Q2_K's and TQ2_0's `qs`).
    fn quarter_bits(byte: u32) -> Vec<Option<u32>> {
        (0..512)
            .map(|bit| {
                let position = bit / 2;
                Some(
                    (byte + (position / 128) * 32 + position % 32) * 8
                        + (position % 128 / 32) * 2
                        + bit % 2,
                )
            })
            .collect()
    }

    /// The bit routes of `entries` packed entries of `bits` bits each, bit
    /// `b` of entry `e` from source bit `route(e, b)` (zero where it is
    /// none).
    fn entry_bits<R: Into<Option<u32>>>(
        entries: u32,
        bits: u32,
        route: impl Fn(u32, u32) -> R,
    ) -> Vec<Option<u32>> {
        (0..entries * bits)
            .map(|bit| route(bit / bits, bit % bits).into())
            .collect()
    }

    fn hierarchical_coefficients() -> Vec<Option<u32>> {
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
            .map(Some)
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
                            .map(Some)
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
                            .map(Some)
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
                            .map(Some)
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
                    PlaneRepackRecipe::BitRoutes(entry_bits(
                        32,
                        crate::repr::IQ3XXS_CODES.entry_bits(),
                        |entry, bit| {
                            let (block, group) = (entry / 4, entry % 4);
                            match bit {
                                0..16 => Some((2 + 8 * block + 2 * group) * 8 + bit),
                                16..23 => Some((66 + 4 * block) * 8 + 7 * group + bit - 16),
                                _ => None,
                            }
                        },
                    )),
                ));
                push(plane(
                    "coefficients",
                    PlaneRepackRecipe::BitRoutes(
                        (0..32)
                            .map(|bit| (66 + 4 * (bit / 4)) * 8 + 28 + bit % 4)
                            .map(Some)
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
            "gguf_iq2_xxs" => {
                // block_iq2_xxs: d (f16) | eight u32 pairs, one per 32
                // values: four 8-bit grid indices of eight values, then four
                // 7-bit sign groups and the 4-bit scale. Entry `e` (values
                // 8e..8e+8 of sub-block e / 4) is
                // `qs[8 (e / 4) + e % 4] | signs (e % 4) << 8`.
                push(plane(
                    "words",
                    PlaneRepackRecipe::BitRoutes(entry_bits(
                        32,
                        crate::repr::IQ2XXS_CODES.entry_bits(),
                        |entry, bit| {
                            let (block, group) = (entry / 4, entry % 4);
                            match bit {
                                0..8 => Some((2 + 8 * block + group) * 8 + bit),
                                8..15 => Some((6 + 8 * block) * 8 + 7 * group + bit - 8),
                                _ => None,
                            }
                        },
                    )),
                ));
                push(plane(
                    "coefficients",
                    PlaneRepackRecipe::BitRoutes(
                        (0..32)
                            .map(|bit| (6 + 8 * (bit / 4)) * 8 + 28 + bit % 4)
                            .map(Some)
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
                // qs[128]. Value w of 32-value block b is the low (w < 16)
                // or high nibble of qs[16 b + w % 16]; block b's six-bit
                // scale code is nibble b % 2 of scales_l[b / 2] under bits
                // 2b, 2b + 1 of scales_h.
                push(plane(
                    "words",
                    PlaneRepackRecipe::BitRoutes(
                        (0..1024)
                            .map(|bit| {
                                let position = bit / 4;
                                let source_byte = 8 + (position / 32) * 16 + position % 16;
                                source_byte * 8 + (position % 32 / 16) * 4 + bit % 4
                            })
                            .map(Some)
                            .collect(),
                    ),
                ));
                push(plane(
                    "coefficients",
                    PlaneRepackRecipe::BitRoutes(entry_bits(8, 6, |block, bit| match bit {
                        0..4 => (4 + block / 2) * 8 + 4 * (block % 2) + bit,
                        _ => 16 + 2 * block + bit - 4,
                    })),
                ));
                push(plane(
                    "scale_factor",
                    PlaneRepackRecipe::BitRoutes(direct_bits(0, 2)),
                ));
            }
            "gguf_q3_k" => {
                // block_q3_K: hmask[32] | qs[64] | scales[12] | d (f16).
                // Value p: low bits `(qs[32 (p / 128) + p % 32] >> 2 (p % 128 / 32)) & 3`,
                // high bit `(hmask[p % 32] >> (p / 32)) & 1`, its code
                // `low + 4 high` (q = code - 4). Sub-block j's six-bit scale
                // code: low four bits `scales[j % 8] >> 4 (j / 8)`, high two
                // bits `scales[8 + j % 4] >> 2 (j / 4)`.
                push(plane(
                    "words",
                    PlaneRepackRecipe::BitRoutes(entry_bits(256, 3, |p, bit| match bit {
                        0..2 => (32 + (p / 128) * 32 + p % 32) * 8 + (p % 128 / 32) * 2 + bit,
                        _ => (p % 32) * 8 + p / 32,
                    })),
                ));
                push(plane(
                    "coefficients",
                    PlaneRepackRecipe::BitRoutes(entry_bits(16, 6, |j, bit| match bit {
                        0..4 => (96 + j % 8) * 8 + 4 * (j / 8) + bit,
                        _ => (104 + j % 4) * 8 + 2 * (j / 4) + bit - 4,
                    })),
                ));
                push(plane(
                    "scale_factor",
                    PlaneRepackRecipe::BitRoutes(direct_bits(108, 2)),
                ));
            }
            "gguf_iq3_s" => {
                // block_iq3_s: d (f16) | qs[64] | qh[8] | signs[32] | scales[4].
                // Entry `e` (values 8e..8e+8, group l = e % 4 of 32-value
                // sub-block b = e / 4) is its two 9-bit indices
                // `qs[8 b + 2 l + i] | ((qh[b] >> (2 l + i)) & 1) << 8`
                // (i = 0, 1), then the sign byte `signs[e]`; scales[b / 2]
                // holds sub-block b's 4-bit scale in nibble b % 2.
                push(plane(
                    "words",
                    PlaneRepackRecipe::BitRoutes(entry_bits(
                        32,
                        crate::repr::IQ3S_CODES.entry_bits(),
                        |entry, bit| {
                            let (block, group) = (entry / 4, entry % 4);
                            match bit {
                                0..18 => {
                                    let (index, bit) = (bit / 9, bit % 9);
                                    if bit < 8 {
                                        (2 + 8 * block + 2 * group + index) * 8 + bit
                                    } else {
                                        (66 + block) * 8 + 2 * group + index
                                    }
                                }
                                _ => (74 + entry) * 8 + bit - 18,
                            }
                        },
                    )),
                ));
                push(plane(
                    "coefficients",
                    PlaneRepackRecipe::BitRoutes(direct_bits(106, 4)),
                ));
                push(plane(
                    "scale_factor",
                    PlaneRepackRecipe::BitRoutes(direct_bits(0, 2)),
                ));
            }
            "gguf_q4_0" | "gguf_iq4_nl" => {
                // block_q4_0 and block_iq4_nl: d (f16) | qs[16]. Value p
                // is the low (p < 16) or high nibble of qs[p % 16]; d moves
                // bit for bit.
                push(plane(
                    "words",
                    PlaneRepackRecipe::BitRoutes(
                        (0..128)
                            .map(|bit| {
                                let position = bit / 4;
                                (2 + position % 16) * 8 + (position / 16) * 4 + bit % 4
                            })
                            .map(Some)
                            .collect(),
                    ),
                ));
                push(plane(
                    "scale",
                    PlaneRepackRecipe::BitRoutes(direct_bits(0, 2)),
                ));
            }
            "gguf_q2_k" => {
                // block_q2_K: scales[16] (a 4-bit scale, then a 4-bit
                // minimum, per 16 values) | qs[64] | d (f16) | dmin (f16).
                push(plane("words", PlaneRepackRecipe::BitRoutes(quarter_bits(16))));
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
            "gguf_tq2_0" => {
                // block_tq2_0: qs[64] | d (f16).
                push(plane("words", PlaneRepackRecipe::BitRoutes(quarter_bits(0))));
                push(plane(
                    "scale",
                    PlaneRepackRecipe::BitRoutes(direct_bits(64, 2)),
                ));
            }
            "gguf_q4_1" => {
                // block_q4_1: d (f16) | m (f16) | qs[16]: value p is
                // d q + m of q the low (p < 16) or high nibble of qs[p % 16];
                // d and m move bit for bit to the scale and the bias.
                push(plane(
                    "words",
                    PlaneRepackRecipe::BitRoutes(
                        (0..128)
                            .map(|bit| {
                                let position = bit / 4;
                                (4 + position % 16) * 8 + (position / 16) * 4 + bit % 4
                            })
                            .map(Some)
                            .collect(),
                    ),
                ));
                push(plane(
                    "scale",
                    PlaneRepackRecipe::BitRoutes(direct_bits(0, 2)),
                ));
                push(plane(
                    "bias",
                    PlaneRepackRecipe::BitRoutes(direct_bits(2, 2)),
                ));
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
            "gguf_f8_e4m3_e8m0_1x32" | "gguf_f8_e4m3_e8m0_32x32" => {
                // The packet: e (E8M0) | qs[32] (E4M3), value p qs[p]; every
                // byte moves bit for bit.
                push(plane("words", PlaneRepackRecipe::BitRoutes(direct_bits(1, 32))));
                push(plane(
                    "block_scale",
                    PlaneRepackRecipe::BitRoutes(direct_bits(0, 1)),
                ));
            }
            "gguf_f8_e4m3_f32_128x128" => {
                // The packet: s (F32) | qs[128] (E4M3), value p qs[p]; every
                // byte moves bit for bit.
                push(plane("words", PlaneRepackRecipe::BitRoutes(direct_bits(4, 128))));
                push(plane("scale", PlaneRepackRecipe::BitRoutes(direct_bits(0, 4))));
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
                            .map(Some)
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
            "gguf_q1_0" => {
                // block_q1_0: d (f16) | qs[16]: value p is bit p % 8 of
                // qs[p / 8], ggml's `±d`.
                push(plane(
                    "words",
                    PlaneRepackRecipe::BitRoutes(direct_bits(2, 16)),
                ));
                push(plane(
                    "scale",
                    PlaneRepackRecipe::BitRoutes(direct_bits(0, 2)),
                ));
            }
            "gguf_q2_0" => {
                // block_q2_0: d (f16) | qs[16]: value p is bits 2 (p % 4) of
                // qs[p / 4], ggml's `(q - 1) d`.
                push(plane(
                    "words",
                    PlaneRepackRecipe::BitRoutes(direct_bits(2, 16)),
                ));
                push(plane(
                    "scale",
                    PlaneRepackRecipe::BitRoutes(direct_bits(0, 2)),
                ));
            }
            "gguf_iq2_s" => {
                // block_iq2_s: d (f16) | qs[32] | signs[32] | qh[8] |
                // scales[8]. Entry `e` (values 8e..8e+8) is its 10-bit index
                // `qs[e] | (qh[e / 4] >> 2 (e % 4) & 3) << 8`, then the sign
                // byte `signs[e]`; scales[i] holds the 4-bit scales of values
                // 32i..32i+16 (low nibble) and 32i+16..32i+32.
                push(plane(
                    "words",
                    PlaneRepackRecipe::BitRoutes(entry_bits(
                        32,
                        crate::repr::IQ2S_CODES.entry_bits(),
                        |entry, bit| match bit {
                            0..8 => (2 + entry) * 8 + bit,
                            8..10 => (66 + entry / 4) * 8 + 2 * (entry % 4) + bit - 8,
                            _ => (34 + entry) * 8 + bit - 10,
                        },
                    )),
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
            "gguf_iq1_s" => {
                // block_iq1_s: d (f16) | qs[32] | qh[8] (u16). Entry `e` is
                // its 11-bit index `qs[e] | (qh[e / 4] >> 3 (e % 4) & 7) << 8`
                // and its carried bit, zero; qh[i]'s bits 12..15 are the
                // scale and the bias sign of values 32i..32i+32, its 4-bit
                // local code.
                push(plane(
                    "words",
                    PlaneRepackRecipe::BitRoutes(entry_bits(
                        32,
                        crate::repr::IQ1S_CODES.entry_bits(),
                        |entry, bit| match bit {
                            0..8 => Some((2 + entry) * 8 + bit),
                            8..11 => Some((34 + 2 * (entry / 4)) * 8 + 3 * (entry % 4) + bit - 8),
                            _ => None,
                        },
                    )),
                ));
                push(plane(
                    "coefficients",
                    PlaneRepackRecipe::BitRoutes(entry_bits(8, 4, |group, bit| {
                        (34 + 2 * group) * 8 + 12 + bit
                    })),
                ));
                push(plane(
                    "scale_factor",
                    PlaneRepackRecipe::BitRoutes(direct_bits(0, 2)),
                ));
            }
            "gguf_iq1_m" => {
                // block_iq1_m: qs[32] | qh[16] | scales[8] (u16 sc[4]). Entry
                // `e` is its 11-bit index `qs[e]` and the low three bits of
                // nibble `e % 2` of qh[e / 2], then that nibble's top bit, its
                // delta. The 3-bit scale of values 16f..16f+16 is bits
                // 3 (f % 4).. of sc[f / 4]; d's nibble `n` is sc[n]'s top
                // nibble (ggml's `iq1m_scale_t`).
                push(plane(
                    "words",
                    PlaneRepackRecipe::BitRoutes(entry_bits(
                        32,
                        crate::repr::IQ1M_CODES.entry_bits(),
                        |entry, bit| match bit {
                            0..8 => entry * 8 + bit,
                            _ => (32 + entry / 2) * 8 + 4 * (entry % 2) + bit - 8,
                        },
                    )),
                ));
                push(plane(
                    "coefficients",
                    PlaneRepackRecipe::BitRoutes(entry_bits(16, 3, |field, bit| {
                        48 * 8 + 16 * (field / 4) + 3 * (field % 4) + bit
                    })),
                ));
                // d moves bit for bit; the representation scales it by 2^-3.
                push(plane(
                    "scale_factor",
                    PlaneRepackRecipe::BitRoutes(
                        (0..16)
                            .map(|bit| 48 * 8 + 16 * (bit / 4) + 12 + bit % 4)
                            .map(Some)
                            .collect(),
                    ),
                ));
            }
            "gguf_tq1_0" => {
                // block_tq1_0: qs[48] | qh[4] | d (f16). Value p is its
                // base-3 digit (`TQ1_DIGITS`), a 2-bit code offset by 1 as
                // TQ2_0's; d moves bit for bit.
                push(plane(
                    "words",
                    PlaneRepackRecipe::PackedEntries(
                        (0..256)
                            .map(|p| {
                                let (byte, place) = TQ1_DIGITS.place(p);
                                RepackExpr::Lookup {
                                    index: Box::new(RepackExpr::SourceBits {
                                        bit: byte * 8,
                                        width: 8,
                                    }),
                                    table: RepackTable::Digit(place),
                                }
                            })
                            .collect(),
                    ),
                ));
                push(plane(
                    "scale",
                    PlaneRepackRecipe::BitRoutes(direct_bits(52, 2)),
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
                        routes.iter().flatten().all(|source| *source < source_bits),
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
                        PlaneEncoding::Packed { .. } | PlaneEncoding::SignedLocal { .. } => {
                            let bits = plane.encoding.entry_bits();
                            (bits, 0, (1i64 << bits) - 1)
                        }
                        PlaneEncoding::Grid(grid) => {
                            (grid.entry_bits(), 0, (1i64 << grid.entry_bits()) - 1)
                        }
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
        tables().capabilities.get(id.position()).unwrap_or_else(|| {
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
        tables().intrinsics.get(id.position()).unwrap_or_else(|| {
            panic!("StaticRegistry produced an IntrinsicId outside its intrinsic table (§13.3.1)")
        })
    }
    pub(super) fn intrinsics(capability: CapabilityId) -> &'static [IntrinsicSignature] {
        let t = tables();
        let (start, end) = *t
            .intrinsic_ranges
            .get(capability.position())
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
        tables().representations.get(id.position()).unwrap_or_else(|| {
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
        &conversion_table().0
    }
    fn conversion_table() -> &'static (Vec<RepresentationConversion>, Positions) {
        let tables = tables();
        tables
            .conversions
            .get_or_init(|| conversions(&tables.representations))
    }
    pub(super) fn representation_conversion_info(
        id: RepresentationConversionId,
    ) -> &'static RepresentationConversion {
        representation_conversions().get(id.position()).unwrap_or_else(|| {
            panic!("StaticRegistry produced a RepresentationConversionId outside its conversion table (§13.3.1)")
        })
    }
    pub(super) fn dense(dtype: DType) -> RepresentationId {
        tables().representations[usize::from(dtype.ordinal())].id
    }
    pub(super) fn position(table: Table, key: u32) -> Option<u32> {
        let t = tables();
        let Positions(positions) = match table {
            Table::Capability => &t.capability_positions,
            Table::Intrinsic => &t.intrinsic_positions,
            Table::Representation => &t.representation_positions,
            Table::RepresentationConversion => &conversion_table().1,
        };
        positions.get(&key).copied()
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
        *tables().denotations.get(id.position()).unwrap_or_else(|| {
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
    #[test]
    fn units_take_their_bytes_by_the_order_their_strides_accumulate() {
        // A zero extent met before an overflow (from the last axis) leaves
        // no bytes; one met after the product overflowed does not.
        assert_eq!(super::units_bytes(&[u64::MAX, 2, 0], 1), Some(0));
        assert_eq!(super::units_bytes(&[0, u64::MAX, 2], 1), None);
        assert_eq!(super::units_bytes(&[3, 4, 5], 2), Some(120));
        assert_eq!(super::units_bytes(&[3, 4, 5], u64::MAX), None);
    }

    /// A chunk's entries are its records' addresses within it, from its
    /// first record; the last chunk's stop at the reservation's last.
    #[test]
    fn a_chunk_enters_its_records_from_its_first() {
        let address = |entries: &[u8]| {
            entries
                .chunks_exact(super::RECORD_TABLE_ENTRY_BYTES as usize)
                .map(|entry| u64::from_le_bytes(entry.try_into().unwrap()))
                .collect::<Vec<_>>()
        };
        // 4 records of 512 bytes a 2 KiB chunk, 10 records in all.
        let (start, entries) = super::chunk_entries(1, 2048, 512, 10, 0x10_0000);
        assert_eq!((start, address(&entries)), (4, vec![0x10_0000, 0x10_0200, 0x10_0400, 0x10_0600]));
        let (start, entries) = super::chunk_entries(2, 2048, 512, 10, 0x20_0000);
        assert_eq!((start, address(&entries)), (8, vec![0x20_0000, 0x20_0200]));
    }

    /// The fragment slots a kernel places from are the inverse of `code_bit`:
    /// every (row, column) of a tile's k16 step, by the lane and slot that
    /// hold it.
    #[test]
    fn fragment_slots_invert_the_code_placement() {
        for r in 0..MMA_TILE_ROWS {
            for k in 0..MMA_STEP {
                let lane = 4 * (r % 8) + (k % 8) / 2;
                let slot = (k / 8) * 2 + r / 8 + 4 * (k % 2);
                assert_eq!(PackedRowLayout::mma_fragment_row(lane, slot), r);
                assert_eq!(PackedRowLayout::mma_fragment_column(lane, slot), k);
            }
        }
    }

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

    /// Every registry id is its entry's key, a function of the entry's own
    /// name alone (`Table::key`), so an entry the registry adds moves no
    /// other's: these keys are what a stored identity of each holds, and
    /// stay as pinned whatever is registered beside them. Each id decodes
    /// back from its key to its own row.
    #[test]
    fn registry_ids_are_keyed_by_their_entries_alone() {
        for info in representations() {
            assert_eq!(info.id.key(), Table::Representation.key(info.name), "{}", info.name);
            assert_eq!(position(Table::Representation, info.id.key()), Some(info.id.position() as u32));
            assert_eq!(representation_info(info.id).name, info.name);
        }
        for conversion in representation_conversions() {
            let name = |id| representation_info(id).name;
            let named = format!("{}->{}", name(conversion.source), name(conversion.destination));
            assert_eq!(conversion.id.key(), Table::RepresentationConversion.key(&named));
            assert_eq!(representation_conversion_info(conversion.id).id, conversion.id);
        }
        for backend in BackendName::ALL {
            for info in capabilities(backend) {
                let named = format!("{}/{}", backend.as_str(), info.name);
                assert_eq!(info.id.key(), Table::Capability.key(&named));
            }
        }
        for (name, key) in [
            ("bf16", 0x3ea6_6b24),
            ("q4k", 0x6fa9_755d),
            ("q4k@rows8", 0x01b9_0318),
            ("q8g32s", 0x3798_44ef),
            ("q8g32s@rows16", 0x22c8_712e),
            ("gguf_q4_k", 0x4ab2_f455),
            ("gguf_q8_0", 0x1f9a_479c),
        ] {
            assert_eq!(representation(name).map(|id| id.key()), Some(key), "{name}");
        }
        assert_eq!(position(Table::Representation, 0), None);
    }

    #[test]
    fn overloads_resolve_each_row_by_its_operands() {
        for backend in BackendName::ALL {
            for info in capabilities(backend) {
                for signature in intrinsics(info.id) {
                    let overload = intrinsic_overloads(info.id, signature.name);
                    assert!(overload.contains(&signature.id));
                    assert!(overload.windows(2).all(|pair| pair[0].position() < pair[1].position()));
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
    /// and keeps every other source bit: IQ2_XS's and IQ3_XXS's factors, and
    /// nothing of Q4_0, whose scale moves bit for bit.
    #[test]
    fn a_conversion_admits_sources_with_finite_converted_scales() {
        for (source, destination, fields) in [
            ("gguf_iq2_xs", "iq2xs@rows16", vec![0]),
            ("gguf_iq3_xxs", "iq3xxs@rows16", vec![0]),
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
            conversion.admit_source(&[2, u64::from(layout.logical_group)], &mut packets);
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

    /// A finite source also makes the scales a conversion carries bit for
    /// bit finite: Q4_0's and Q8_0's binary16 `d` (bytes 0 and 1) and
    /// MXFP4's E8M0 exponent (byte 0), every other byte kept; IQ2_XS's
    /// converted factor as its domain does.
    #[test]
    fn a_finite_source_makes_every_scale_its_destination_reads_finite() {
        for (source, destination, cleared) in [
            ("gguf_q4_0", "q4g32s@rows16", (1, 0xbf)),
            ("gguf_q8_0", "q8g32s@rows16", (1, 0xbf)),
            ("gguf_iq2_xs", "iq2xs@rows16", (1, 0xbf)),
            ("gguf_mxfp4", "mxfp4g32@rows16", (0, 0x7f)),
        ] {
            let conversion = representation_conversion(
                representation(source).unwrap(),
                representation(destination).unwrap(),
            )
            .unwrap();
            let RepresentationKind::External(layout) = &representation_info(conversion.source).kind
            else {
                panic!("{source} is external storage")
            };
            let mut packets = vec![0xffu8; 2 * layout.packet_size as usize];
            conversion.finite_source(&[2, u64::from(layout.logical_group)], &mut packets);
            for (packet, bytes) in packets.chunks_exact(layout.packet_size as usize).enumerate() {
                for (index, byte) in bytes.iter().enumerate() {
                    let expected = if index == cleared.0 { cleared.1 } else { 0xff };
                    assert_eq!(*byte, expected, "{source} packet {packet} byte {index}");
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
            plane_set("q3k@rows16"),
            [
                ("codes_lo", 64),
                ("codes_hi", 32),
                ("scales", 12),
                ("supers", 2)
            ]
        );
        assert_eq!(
            plane_set("iq4xs@rows16"),
            [("codes_lo", 128), ("scales", 6), ("supers", 2)]
        );
        assert_eq!(
            plane_set("iq3s@rows16"),
            [
                ("grids", 96),
                ("grid_high", 8),
                ("scales", 4),
                ("supers", 2)
            ]
        );
        // Qwen3.5 K = 2560: codes 1280 | scales 120 -> 128 | supers 40 -> 48.
        let q4k = row_layout("q4k@rows16");
        assert_eq!(q4k.plane_row_offset(1, 2560u64), Some(1280));
        assert_eq!(q4k.plane_row_offset(2, 2560u64), Some(1408));
        assert_eq!(q4k.row_stride_bytes(2560u64), Some(1456));
        assert_eq!(
            canonical_bytes(representation("q4k@rows16").unwrap(), &[3u64, 2560]),
            Some(3 * 1456)
        );
        // A q8 row of three groups: 96 code bytes, 6 super bytes -> 16.
        let q8 = row_layout("q8g32s@rows16");
        assert_eq!(q8.row_stride_bytes(96u64), Some(112));
        // mma16 pads rows per matrix and q8 rows to whole 64-column k-blocks.
        let q8_mma = representation("q8g32s@mma16").unwrap();
        assert_eq!(
            row_layout("q8g32s@mma16").row_stride_bytes(96u64),
            Some(128 + 16)
        );
        assert_eq!(canonical_bytes(q8_mma, &[2u64, 17, 96]), Some(2 * 32 * 144));
        assert_eq!(canonical_bytes(q8_mma, &[96u64]), None);
        // mma16 rows hold whole k-blocks and whole 16-byte chunks of every
        // code plane: a 32-value group's one-bit high plane (4 bytes) takes
        // groups in fours, so Gemma's K = 704 stores 24 groups.
        for (name, multiple) in [
            ("q4k@mma16", 1),
            ("q6k@mma16", 1),
            ("q8g32s@mma16", 2),
            ("q4g32s@mma16", 2),
            ("mxfp4g32@mma16", 2),
            ("nvfp4g16@mma16", 1),
            ("q5g32s@mma16", 4),
            ("q5g32@mma16", 4),
        ] {
            assert_eq!(row_layout(name).group_multiple, multiple, "{name}");
        }
        assert_eq!(row_layout("q5g32@mma16").row_groups(704u64), Some(24));
        assert_eq!(row_layout("q5g32@rows16").row_groups(704u64), Some(22));
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
        assert_eq!(layout.row_stride_bytes(4096u64), Some(1056));
        assert_eq!(layout.row_stride_bytes(5376u64), Some(1392));
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

    /// A refused store's diagnostic is its rule's: packed storage takes
    /// whole records, external storage no store at all.
    #[test]
    fn a_refused_store_reports_its_rule() {
        let packed = representation("q4g64").unwrap();
        let external = representation("gguf_q4_k").unwrap();
        assert_eq!(store_rule(dense(DType::F32)).admits(false), Ok(()));
        assert_eq!(store_rule(packed).admits(true), Ok(()));
        assert!(store_rule(packed).admits(false).unwrap_err().contains("decode-only"));
        assert!(store_rule(packed).admits_value(None).unwrap_err().contains("decode-only"));
        assert_eq!(store_rule(external), StoreRule::Refused(external));
        assert!(store_rule(external).admits(true).unwrap_err().contains("takes no store"));
        assert!(!record_writable(external));
    }

    #[test]
    fn canonical_bytes_follow_the_representation_layout() {
        let q4g64 = representation("q4g64").unwrap();
        let q4_k = representation("gguf_q4_k").unwrap();
        assert_eq!(canonical_bytes(dense(DType::F32), &[3u64, 5]), Some(60));
        assert_eq!(canonical_bytes(dense(DType::BF16), &[]), Some(2));
        assert_eq!(canonical_bytes(dense(DType::F32), &[0u64, 5]), Some(0));
        assert_eq!(canonical_bytes(dense(DType::F32), &[u64::MAX, 2, 0]), Some(0));
        assert_eq!(canonical_bytes(dense(DType::F32), &[0, u64::MAX, 2]), None);
        assert_eq!(canonical_bytes(q4g64, &[3u64, 128]), Some(3 * 2 * 36));
        assert_eq!(canonical_bytes(q4g64, &[2u64, 3, 65]), Some(6 * 2 * 36));
        assert_eq!(canonical_bytes(q4_k, &[512u64]), Some(288));
        assert_eq!(canonical_bytes(q4_k, &[2u64, 257]), Some(2 * 2 * 144));
        assert_eq!(canonical_bytes(q4g64, &[]), None);
        assert_eq!(canonical_bytes(q4_k, &[]), None);
        assert_eq!(canonical_bytes(dense(DType::F32), &[u64::MAX, 2]), None);
        assert_eq!(canonical_bytes(q4g64, &[u64::MAX, 64]), None);
    }

    /// A record is a leading-axis row in its own bytes: a matrix of a row
    /// layout's tensor of matrices, never a row its tiles interleave.
    #[test]
    fn records_are_leading_rows_in_their_own_bytes() {
        let bytes = |id, extents: &[u64]| record_bytes(id, extents).map(std::num::NonZeroU64::get);
        let q4_k = representation("gguf_q4_k").unwrap();
        assert_eq!(bytes(dense(DType::F32), &[3, 5]), Some(20));
        assert_eq!(bytes(dense(DType::F32), &[0, 5]), Some(20), "a view of no records has records of a size");
        assert_eq!(bytes(dense(DType::F32), &[3, 0]), None, "empty records have none");
        assert_eq!(bytes(q4_k, &[2, 257]), Some(2 * 144));
        assert_eq!(bytes(q4_k, &[512]), None);
        for layout in [Layout::Rows16, Layout::Rows8, Layout::Mma16, Layout::Rows32] {
            let id = storage("q4k", layout).unwrap();
            let matrix = canonical_bytes(id, &[24u64, 256]).unwrap();
            assert_eq!(bytes(id, &[3, 24, 256]), Some(matrix), "{}", layout.as_str());
            assert_eq!(canonical_bytes(id, &[3u64, 24, 256]), Some(3 * matrix));
            let RepresentationKind::PackedRows(rows) = &representation_info(id).kind else {
                unreachable!("a row layout's storage")
            };
            let row = (rows.tile_rows() == 1).then(|| canonical_bytes(id, &[256u64]).unwrap());
            assert_eq!(bytes(id, &[32, 256]), row, "{}", layout.as_str());
        }
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
            assert_eq!(Some(canonical.len() as u64), canonical_bytes(info.id, &[rows, extent]));
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

    /// F8_E4M3 has a source only under a registered grid of block scales,
    /// one source a grid, named for its scale and tile: a packet the scale
    /// then a group of codes, converting under E8M0 scales (one byte, 32
    /// codes) into mxfp8g32, under F32 scales (four bytes, 128 codes) into
    /// fp8g128, whose decoders read E4M3 bytes as their values (the
    /// profile's integers under exponent -9) under that scale; no other
    /// encoding is read under a grid, and no other tile is a source.
    #[test]
    fn f8_e4m3_is_sourced_by_its_scale_grid() {
        use crate::repr::ValueCodes;
        assert_eq!(gguf_source(Encoding::F8E4M3, GgufScales::Own, 256), None);
        let unregistered = Grid { scale: GridScale::F32, tile: [32, 128] };
        assert_eq!(gguf_source(Encoding::F8E4M3, GgufScales::Grid(unregistered), 256), None);
        for (grid, name, resident, packet_size, group, encoding) in [
            (Grid::REGISTERED[0], "gguf_f8_e4m3_e8m0_1x32", "mxfp8g32", 33, 32, ScaleEncoding::E8M0),
            (Grid::REGISTERED[1], "gguf_f8_e4m3_e8m0_32x32", "mxfp8g32", 33, 32, ScaleEncoding::E8M0),
            (Grid::REGISTERED[2], "gguf_f8_e4m3_f32_128x128", "fp8g128", 132, 128, ScaleEncoding::F32),
        ] {
            let scales = GgufScales::Grid(grid);
            assert_eq!(gguf_source(Encoding::F8E4M3, scales, 48), None);
            assert_eq!(gguf_source(Encoding::Q8_0, scales, 256), None);
            let source = gguf_source(Encoding::F8E4M3, scales, 256).unwrap();
            assert_eq!(representation_info(source).name, name);
            let packet = external_packet(source).unwrap();
            assert_eq!((packet.packet_size, packet.logical_group), (packet_size, group));
            let tile = grid.tile.map(|extent| extent as u32);
            assert_eq!(packet.planes, Some(GridPlanes { scale_bytes: grid.scale.bytes() as u32, tile }));
            for layout in Layout::ALL {
                let element = conversion_destination(source, layout, OperandRecipe::Rounded).unwrap();
                assert_eq!(representation_info(element).representation, resident);
                if let Some(axes) = decode_axes(element) {
                    assert_eq!((axes.code_exponent, axes.profile().code_exponent), (0, -9));
                    let CodingAxes::Direct { codes, scale, bias, group: coded } = axes.coding else {
                        panic!("{resident} is direct");
                    };
                    assert_eq!((codes, scale, bias, coded), (ValueCodes::E4m3, encoding, false, group as u32));
                }
            }
        }
    }

    /// A grid source is planar, each matrix its codes then its grid, as a
    /// file stores them: a packet's scale is its tile's in the grid after
    /// the matrix's codes, its codes its row's in the codes plane, and a
    /// tensor's bytes are its matrices', a matrix of `[N, K]` `N·K` code
    /// bytes and `N/r · K/c` scales. A source of its own scales is its
    /// packets one after another.
    #[test]
    fn a_grid_source_addresses_its_packets_in_its_planes() {
        let f32 = external_packet(representation("gguf_f8_e4m3_f32_128x128").unwrap()).unwrap();
        let (rows, columns) = (256u64, 512u64);
        let codes = rows * columns;
        assert_eq!(f32.matrix_bytes(rows, columns), Some(codes + 2 * 4 * 4));
        // A matrix has rows: a planar tensor of one axis is no tensor.
        assert!(f32.matrices(&[columns]).is_none());
        assert_eq!(f32.matrices(&[rows, columns]).map(|matrices| matrices.count), Some(1));
        // Row 130, packet 3: codes at row 130's fourth group; its scale the
        // grid's row 1 (rows 128..256), column 3 (values 384..512).
        assert_eq!(
            f32.packet_pieces(rows, columns, 130, 3),
            PacketPieces { codes: 130 * columns + 3 * 128, scale: Some(codes + (4 + 3) * 4) }
        );
        assert_eq!(f32.packet_byte(rows, columns, 130, 3, 2), codes + 7 * 4 + 2);
        assert_eq!(f32.packet_byte(rows, columns, 130, 3, 4), 130 * columns + 3 * 128);
        let source = representation("gguf_f8_e4m3_f32_128x128").unwrap();
        assert_eq!(canonical_bytes(source, &[3, rows, columns]), Some(3 * (codes + 32)));
        assert_eq!(record_bytes(source, &[3, rows, columns]).map(|bytes| bytes.get()), Some(codes + 32));
        assert_eq!(record_bytes(source, &[rows, columns]), None, "a planar matrix's rows share its grid");
        // Per row of 32: packet 5 of row 9 takes scale 9 · 16 + 5.
        let row = external_packet(representation("gguf_f8_e4m3_e8m0_1x32").unwrap()).unwrap();
        assert_eq!(row.matrix_bytes(64, 512), Some(64 * 512 + 64 * 16));
        assert_eq!(row.packet_byte(64, 512, 9, 5, 0), 64 * 512 + 9 * 16 + 5);
        assert_eq!(row.packet_byte(64, 512, 9, 5, 1), 9 * 512 + 5 * 32);
        let q8 = external_packet(representation("gguf_q8_0").unwrap()).unwrap();
        assert_eq!(q8.planes, None);
        assert_eq!(q8.packet_byte(4, 64, 3, 1, 5), (3 * 2 + 1) * 34 + 5);
    }

    /// The host reference converts a grid source as its packets gathered
    /// byte by byte where the registry addresses them, over stacked
    /// matrices of several tile rows (a partial last); a source's admission
    /// clears a packet's bits where they lie, a scale's in the grid; and a
    /// source of one plane's packets are its bytes.
    #[test]
    fn a_grid_source_converts_as_its_gathered_packets() {
        for source in ["gguf_f8_e4m3_e8m0_1x32", "gguf_f8_e4m3_e8m0_32x32", "gguf_f8_e4m3_f32_128x128"] {
            let source = representation(source).unwrap();
            let layout = external_packet(source).unwrap();
            let (matrices, rows, columns) = (2u64, 133u64, 512u64);
            let extents = [matrices, rows, columns];
            let tensor = layout.matrices(&extents).unwrap();
            assert_eq!(Some(tensor.tensor_bytes()), canonical_bytes(source, &extents));
            let bytes = pseudo_random_bytes(source.position() as u64, tensor.tensor_bytes() as usize);
            let gathered = (0..matrices * rows)
                .flat_map(|row| {
                    (0..tensor.row_packets()).flat_map(move |packet| {
                        (0..layout.packet_size).map(move |byte| {
                            row / rows * tensor.bytes + layout.packet_byte(rows, columns, row % rows, packet, byte)
                        })
                    })
                })
                .map(|at| bytes[at as usize])
                .collect::<Vec<_>>();
            assert_eq!(&*tensor.packets(&bytes), &gathered[..]);
            let shape = extents.map(|extent| extent as usize);
            for conversion in representation_conversions().iter().filter(|conversion| conversion.source == source) {
                assert_eq!(
                    crate::interp::repack(conversion.id, &shape, &bytes).unwrap(),
                    crate::interp::repack_packets(conversion.id, &extents, &gathered),
                    "{}",
                    representation_info(conversion.destination).name
                );
                // A finite source clears bits of the carried scales where
                // they lie in each grid, and of a code only where it is NaN
                // (E4M3's S.1111.111), its lowest bit: every finite code
                // kept.
                let mut finite = bytes.clone();
                conversion.finite_source(&extents, &mut finite);
                assert!(finite.iter().zip(&bytes).all(|(finite, byte)| finite & !byte == 0));
                let codes = rows * tensor.row_packets() * u64::from(layout.code_bytes());
                for matrix in 0..matrices {
                    let codes = (matrix * tensor.bytes) as usize..(matrix * tensor.bytes + codes) as usize;
                    for (finite, byte) in finite[codes.clone()].iter().zip(&bytes[codes]) {
                        let expected = if byte & 0x7f == 0x7f { byte & !1 } else { *byte };
                        assert_eq!(*finite, expected, "only a NaN code changes");
                    }
                }
            }
        }
        let q8 = representation("gguf_q8_0").unwrap();
        let tensor = external_packet(q8).unwrap().matrices(&[2, 3, 64]).unwrap();
        let bytes = pseudo_random_bytes(0, tensor.tensor_bytes() as usize);
        assert!(matches!(tensor.packets(&bytes), std::borrow::Cow::Borrowed(packets) if packets == &bytes[..]));
    }
}

/// How a setup run (`SEISMIC_TEST_SETUP`, [`test_setup_only`]) ends a test
/// that formed its setup without a device: the test stops there, its
/// device work unrun, with this message and what it checked.
pub const TEST_SETUP_CHECKED: &str = "seismic test setup: checked";
/// How a setup run ends a test that reached a device ([`test_device_reached`]):
/// nothing was checked of what it does there.
pub const TEST_SETUP_UNCHECKED: &str = "seismic test setup: unchecked";

/// Whether the test runner runs each test's setup alone
/// (`SEISMIC_TEST_SETUP` set), the one owner of that variable: a test then
/// forms what it would prepare and checks it without a device
/// (`seismic_native_tests::setup`), and no test reaches one
/// ([`test_device_reached`]).
pub fn test_setup_only() -> bool {
    std::env::var_os("SEISMIC_TEST_SETUP").is_some()
}

/// Where a test reaches a device (discovery, or a backend's own test
/// device): in a setup run ([`test_setup_only`]) the test stops here,
/// unchecked ([`TEST_SETUP_UNCHECKED`]), before any device is touched.
pub fn test_device_reached(what: &str) {
    if test_setup_only() {
        panic!("{TEST_SETUP_UNCHECKED}: it {what} outside a setup, which a setup run never does");
    }
}

/// The backend the test runner requests (`SEISMIC_TEST_BACKEND`), the one
/// owner of that variable: a test then runs on it alone, and is skipped
/// where it lies outside the test's backends. None when the variable is
/// unset; a value that names no backend panics.
pub fn requested_test_backend() -> Option<BackendName> {
    let name = match std::env::var("SEISMIC_TEST_BACKEND") {
        Ok(name) => name,
        Err(std::env::VarError::NotPresent) => return None,
        Err(error) => panic!("invalid SEISMIC_TEST_BACKEND: {error}"),
    };
    Some(match name.as_str() {
        "cpu" => BackendName::Cpu,
        "metal" => BackendName::Metal,
        "cuda" => BackendName::Cuda,
        "vulkan" => BackendName::Vulkan,
        _ => panic!("unknown SEISMIC_TEST_BACKEND {name:?}"),
    })
}
