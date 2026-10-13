//! Storage lattices: stepping stored values to their adjacent representable
//! values, on the byte encodings the engine uploads.
//!
//! A stored value can only be another representable value of its storage, so
//! a perturbation at storage resolution is realised on the storage lattice:
//! every stored value independently stays, moves to the adjacent
//! representable value above, or moves to the adjacent representable value
//! below. [`Dither`] carries the two move probabilities and the seed;
//! [`Lattice`] describes what a buffer stores; [`Lattice::perturb`] rewrites
//! the buffer in place and returns one [`Tally`] per stored plane.
//!
//! # Lattices
//!
//! Adjacency is always by decoded value, never by a fixed unit in the last
//! place or by code order.
//!
//! - **Dense floating storage** ([`Lattice::dense`]; `f32`, `f16`, `bf16`).
//!   The lattice is the finite values of the dtype. The payload of a binary
//!   interchange format orders magnitudes, so the adjacent value is one
//!   payload step, and the spacing above and below a binade boundary differ.
//!   Either zero moves up to the smallest positive subnormal and down to the
//!   smallest negative subnormal. A smallest subnormal moves toward zero to
//!   the zero of its own sign. The largest finite magnitude has no adjacent
//!   value beyond it: that move is blocked, and an infinity is never
//!   produced. Infinities and NaNs are off the lattice and never move. The
//!   width and the infinity payload come from the dtype and the crate's own
//!   quantizer; no per-format constant is written here.
//! - **Integer code planes** ([`Lattice::codes`]): codes of a fixed width
//!   packed little-endian into `u32` words (code `i` at bit
//!   `bits * (i % (32 / bits))` of word `i / (32 / bits)`, the layout of the
//!   affine K8/V4 history codec), with a declared inclusive range of
//!   meaningful codes. The adjacent value is the adjacent code; a move past
//!   either end of the range is blocked. A stored code outside the range is
//!   off the lattice and never moves.
//! - **Registry-packed representations** ([`Lattice::packed`]). Every plane
//!   of the representation's packet form is stepped on its own lattice, read
//!   from the plane's registry [`PlaneEncoding`]:
//!   a dense plane (a stored scale, bias or factor) uses the floating lattice
//!   of its dtype; a packed integer plane and a floating-code plane use a
//!   level table. The level table is built by decoding every raw code of the
//!   plane's width with the registry's reference recipe
//!   ([`reference_math::code_recipe`], [`reference_math::float_code_recipe`])
//!   and ordering the distinct finite decoded values. The adjacent level of a
//!   code is the next distinct value in that order, so two's-complement,
//!   offset, table-valued and sign-magnitude floating codes all step by
//!   value. When several raw codes decode to the target value (the two zeros
//!   of a sign-magnitude code, a code whose sign bit the format ignores), the
//!   lowest raw code is stored. A raw code that decodes to a non-finite value
//!   is off the lattice. A plane of vector codes (one entry codes several
//!   values through a grid) has no adjacent entry by value: its entries
//!   stay, every draw blocked. Nothing here names a representation, a plane
//!   or a code format, so a representation added to the registry is covered.
//!
//! # Layouts
//!
//! [`Lattice::packed`] operates on the canonical bytes of both the `packet`
//! layout and every resident row layout (`rows16`, `rows8`, `mma16`,
//! `rows32`). The registry addresses every bit of every packet-form plane in
//! a row layout ([`PackedRowLayout::packet_bit`]), so one plane description,
//! the packet form's, serves every layout. An entry is addressed once per
//! run of consecutive bits the layout keeps together: the registry's row
//! plane contents ([`RowPlaneContent`]) say that a row layout splits each
//! code across its code planes and keeps every other plane's bytes together.
//! Only entries that some logical
//! value reads are stored values: entries of a partial last packet beyond
//! the logical extent, padding groups and padding rows are never drawn and
//! never change. External (conversion-source) representations are not
//! covered: the registry gives them a packet size and no plane schema.
//!
//! # Determinism
//!
//! One draw belongs to one stored value. It is a function of the seed, the
//! plane's ordinal in the packet form, and the value's position, and of
//! nothing else, so a buffer receives the same perturbation however it is
//! split into chunks and in whatever order or thread the chunks are
//! processed. The position of a dense element or a code is its index. The
//! position of a packed entry is
//! `(row * packets_per_row + packet) * entries_per_packet + entry`, with
//! `row` the logical row (row-major over the leading extents): it does not
//! depend on the layout, so the same seed moves the same logical entries of a
//! representation in every layout.
//!
//! A chunk ([`Lattice::chunk_bytes`]) is the smallest run of bytes that holds
//! whole stored values: one dense element, one `u32` code word, one packet
//! of the `packet` layout, or one row tile of a row layout.

use crate::ids::RepresentationId;
use crate::reference_math::{self, ReferenceScalar, conversion};
use crate::registry::{
    self, CodeInterpretation, PackedPacketLayout, PackedRowLayout, PlaneEncoding,
    RepresentationKind, RowGeometry, RowPlaneContent,
};
use crate::types::DType;
use std::fmt;
use std::ops::RangeInclusive;

#[cfg(test)]
mod tests;

/// Widest code whose level table is enumerated.
const ENUMERABLE_CODE_BITS: u32 = 16;

/// SplitMix64 increment and finalizer: draw `n` of a stream is the finalizer
/// of `stream + n * GAMMA`, which makes every draw addressable by position.
const GAMMA: u64 = 0x9e37_79b9_7f4a_7c15;

fn mix(mut z: u64) -> u64 {
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

/// Why a lattice or a dither cannot be formed.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum LatticeError {
    /// The move probabilities are not both in `[0, 1]` with a sum of at most 1.
    Probabilities,
    /// Dense storage of a dtype that is not floating.
    NotFloating(DType),
    /// A code width that is zero or does not divide the 32 bits of a word.
    CodeWidth(u32),
    /// A code range that is empty or exceeds the code width.
    CodeRange,
    /// A representation that is dense or external, named by its storage name.
    NotPacked(&'static str),
    /// Extents with no canonical, host-addressable, nonempty byte count in
    /// the representation.
    Geometry,
    /// A plane the registry gives no lattice: a dense plane of a
    /// non-floating dtype, or codes too wide to enumerate.
    Plane {
        representation: &'static str,
        plane: &'static str,
    },
}

impl fmt::Display for LatticeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Probabilities => {
                f.write_str("move probabilities must lie in [0, 1] and sum to at most 1")
            }
            Self::NotFloating(dtype) => {
                write!(f, "dense `{dtype}` storage has no floating lattice")
            }
            Self::CodeWidth(bits) => write!(f, "{bits}-bit codes do not pack into `u32` words"),
            Self::CodeRange => f.write_str("the code range is empty or exceeds the code width"),
            Self::NotPacked(name) => write!(f, "`{name}` is not a packed representation"),
            Self::Geometry => f.write_str(
                "the extents have no canonical nonempty byte count in the representation",
            ),
            Self::Plane {
                representation,
                plane,
            } => write!(f, "plane `{plane}` of `{representation}` has no lattice"),
        }
    }
}

impl std::error::Error for LatticeError {}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Direction {
    Down,
    Up,
}

/// The seeded, position-addressed choice of each stored value's move.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Dither {
    seed: u64,
    /// A 63-bit draw below this moves up.
    up: u64,
    /// A 63-bit draw below this and not below `up` moves down.
    moved: u64,
}

impl Dither {
    /// Each stored value moves up with probability `up`, down with
    /// probability `down`, and otherwise stays. Probabilities are realised to
    /// a resolution of `2^-63`. Distinct stored tensors need distinct seeds.
    pub fn new(seed: u64, up: f64, down: f64) -> Result<Dither, LatticeError> {
        if !(up >= 0.0 && down >= 0.0 && up + down <= 1.0) {
            return Err(LatticeError::Probabilities);
        }
        const DRAWS: f64 = (1u64 << 63) as f64;
        let up = (up * DRAWS) as u64;
        Ok(Dither {
            seed,
            up,
            moved: up + (down * DRAWS) as u64,
        })
    }

    /// The draw stream of the plane at `ordinal`.
    fn stream(&self, ordinal: usize) -> u64 {
        mix(mix(self.seed).wrapping_add((ordinal as u64 + 1).wrapping_mul(GAMMA)))
    }

    fn draw(&self, stream: u64, position: u64) -> Option<Direction> {
        let draw = mix(stream.wrapping_add(position.wrapping_mul(GAMMA))) >> 1;
        if draw >= self.moved {
            None
        } else if draw < self.up {
            Some(Direction::Up)
        } else {
            Some(Direction::Down)
        }
    }
}

/// The realised moves of the stored values of one plane.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Tally {
    /// Stored values visited. Those in no other count drew no move.
    pub values: u64,
    /// Values moved to the adjacent representable value above.
    pub up: u64,
    /// Values moved to the adjacent representable value below.
    pub down: u64,
    /// Values that drew a move and stayed because no representable value is
    /// adjacent on that side: the end of a code range or level table, or the
    /// largest finite magnitude of a floating dtype.
    pub blocked: u64,
    /// Values that drew a move and stayed because the stored value is not on
    /// the lattice: a non-finite float, or a code outside the declared range.
    pub off_lattice: u64,
}

impl std::ops::AddAssign for Tally {
    fn add_assign(&mut self, other: Tally) {
        self.values += other.values;
        self.up += other.up;
        self.down += other.down;
        self.blocked += other.blocked;
        self.off_lattice += other.off_lattice;
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Step {
    To(u32),
    Blocked,
    OffLattice,
}

/// How the raw entries of one plane step to their adjacent values.
#[derive(Clone, Debug)]
enum Steps {
    /// A binary interchange floating payload: `sign` is its sign bit and
    /// `infinity` the payload of positive infinity.
    Float { sign: u32, infinity: u32 },
    /// Unsigned codes meaningful in `min..=max`.
    Range { min: u32, max: u32 },
    /// For every raw code, the step down and the step up.
    Levels(Vec<[Step; 2]>),
    /// Vector codes: an entry codes several values at once and has no
    /// adjacent entry by value, so every move is blocked.
    Fixed,
}

impl Steps {
    fn float(dtype: DType) -> Option<Steps> {
        dtype.is_float().then(|| Steps::Float {
            sign: 1 << (dtype.bytes() * 8 - 1),
            infinity: conversion::quantize(dtype, f64::INFINITY),
        })
    }

    /// The level table of `bits`-wide codes whose exact decoded value is
    /// `decoded`. Levels are the distinct finite decoded values; a level is
    /// stored as the lowest raw code that decodes to it.
    fn levels(bits: u32, decoded: impl Fn(u32) -> f64) -> Steps {
        let values: Vec<f64> = (0..1u32 << bits).map(decoded).collect();
        let mut finite: Vec<u32> = (0..1u32 << bits)
            .filter(|raw| values[*raw as usize].is_finite())
            .collect();
        finite.sort_by(|a, b| {
            values[*a as usize]
                .partial_cmp(&values[*b as usize])
                .expect("finite values are ordered")
                .then(a.cmp(b))
        });
        let mut levels: Vec<u32> = finite.clone();
        levels.dedup_by(|later, first| values[*later as usize] == values[*first as usize]);
        let mut steps = vec![[Step::OffLattice; 2]; values.len()];
        let mut level = 0;
        for raw in finite {
            if values[raw as usize] != values[levels[level] as usize] {
                level += 1;
            }
            let neighbour = |index: Option<usize>| {
                index
                    .and_then(|index| levels.get(index))
                    .map_or(Step::Blocked, |raw| Step::To(*raw))
            };
            steps[raw as usize] = [neighbour(level.checked_sub(1)), neighbour(Some(level + 1))];
        }
        Steps::Levels(steps)
    }

    /// The lattice of one plane of a packed representation, or `None` when
    /// the registry gives the plane none.
    fn of(encoding: &PlaneEncoding) -> Option<Steps> {
        match encoding {
            PlaneEncoding::Dense(dtype) => Steps::float(*dtype),
            PlaneEncoding::Packed {
                interpretation: CodeInterpretation::Grid(_),
                ..
            } => Some(Steps::Fixed),
            PlaneEncoding::Packed {
                bits,
                interpretation,
            } => (*bits <= ENUMERABLE_CODE_BITS).then(|| {
                let recipe = reference_math::code_recipe(interpretation, *bits);
                Steps::levels(*bits, |raw| {
                    reference_math::evaluate(&recipe, &[ReferenceScalar::U32(raw)])
                        .expect("registry code interpretation is total")
                        .to_f64()
                })
            }),
            PlaneEncoding::FloatCode { format } => {
                (format.bits() <= ENUMERABLE_CODE_BITS).then(|| {
                    let recipe = reference_math::float_code_recipe(*format);
                    Steps::levels(format.bits(), |raw| {
                        reference_math::evaluate(&recipe, &[ReferenceScalar::U32(raw)])
                            .expect("floating code interpretation is total")
                            .to_f64()
                    })
                })
            }
        }
    }

    fn step(&self, raw: u32, direction: Direction) -> Step {
        match self {
            Steps::Float { sign, infinity } => {
                let magnitude = raw & !sign;
                if magnitude >= *infinity {
                    Step::OffLattice
                } else if magnitude == 0 {
                    Step::To(match direction {
                        Direction::Up => 1,
                        Direction::Down => sign | 1,
                    })
                } else if (direction == Direction::Up) == (raw & sign != 0) {
                    Step::To(raw - 1)
                } else if magnitude + 1 == *infinity {
                    Step::Blocked
                } else {
                    Step::To(raw + 1)
                }
            }
            Steps::Range { min, max } => {
                if raw < *min || raw > *max {
                    Step::OffLattice
                } else {
                    match direction {
                        Direction::Up if raw < *max => Step::To(raw + 1),
                        Direction::Down if raw > *min => Step::To(raw - 1),
                        Direction::Up | Direction::Down => Step::Blocked,
                    }
                }
            }
            Steps::Levels(steps) => steps[raw as usize][direction as usize],
            Steps::Fixed => Step::Blocked,
        }
    }

    /// Step one entry stored as `runs`; `bit_of(bit)` is the buffer bit
    /// holding bit `bit` of the entry.
    fn apply(
        &self,
        bytes: &mut [u8],
        runs: &[Run],
        bit_of: impl Fn(u32) -> usize,
        direction: Direction,
        tally: &mut Tally,
    ) {
        // Runs partition an entry of at most 32 bits.
        let mut at = [0usize; 32];
        let mut raw = 0u32;
        for (at, run) in at.iter_mut().zip(runs) {
            *at = bit_of(run.first);
            raw |= (window(bytes, *at, run.bits).0 as u32) << run.first;
        }
        match self.step(raw, direction) {
            Step::To(next) => {
                for (at, run) in at.iter().zip(runs) {
                    let (_, mask, span) = window(bytes, *at, run.bits);
                    let shift = at % 8;
                    let value = u64::from(next >> run.first) & mask;
                    for (index, byte) in bytes[at / 8..at / 8 + span].iter_mut().enumerate() {
                        let clear = ((mask << shift) >> (8 * index)) as u8;
                        *byte = (*byte & !clear) | ((value << shift) >> (8 * index)) as u8;
                    }
                }
                match direction {
                    Direction::Up => tally.up += 1,
                    Direction::Down => tally.down += 1,
                }
            }
            Step::Blocked => tally.blocked += 1,
            Step::OffLattice => tally.off_lattice += 1,
        }
    }
}

/// The rounding cells of a dense floating dtype: around every finite value,
/// the reals that round to it.
#[derive(Clone, Debug)]
pub struct RoundingCells {
    dtype: DType,
    steps: Steps,
}

impl RoundingCells {
    pub fn of(dtype: DType) -> Result<RoundingCells, LatticeError> {
        Steps::float(dtype)
            .map(|steps| RoundingCells { dtype, steps })
            .ok_or(LatticeError::NotFloating(dtype))
    }

    pub fn dtype(&self) -> DType {
        self.dtype
    }

    /// The width of the rounding cell of the value with payload `bits`:
    /// half the gap to the adjacent representable value below plus half the
    /// gap to the adjacent representable value above, by the dtype's own
    /// adjacency. The largest finite magnitude has no adjacent value beyond
    /// it; round-to-nearest overflows only from half its inward gap beyond
    /// it, so its cell extends that far outward. None for a payload that is
    /// not finite.
    pub fn width(&self, bits: u32) -> Option<f64> {
        let value = conversion::exact_f64(self.dtype, bits);
        let gap = |direction| match self.steps.step(bits, direction) {
            Step::To(adjacent) => Some(Some(
                (conversion::exact_f64(self.dtype, adjacent) - value).abs(),
            )),
            Step::Blocked => Some(None),
            Step::OffLattice => None,
        };
        let (below, above) = (gap(Direction::Down)?, gap(Direction::Up)?);
        let inward = below
            .or(above)
            .expect("a floating dtype has two finite values");
        Some(0.5 * below.unwrap_or(inward) + 0.5 * above.unwrap_or(inward))
    }
}

/// Bits `[first, first + bits)` of an entry, held in consecutive buffer bits
/// (little-endian within bytes).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Run {
    first: u32,
    bits: u32,
}

/// The `bits`-bit value at buffer bit `first`, its mask, and the bytes it
/// spans.
fn window(bytes: &[u8], first: usize, bits: u32) -> (u64, u64, usize) {
    let shift = first % 8;
    let span = (shift + bits as usize).div_ceil(8);
    let mask = (1u64 << bits) - 1;
    let value = bytes[first / 8..first / 8 + span]
        .iter()
        .enumerate()
        .fold(0u64, |value, (index, byte)| {
            value | (u64::from(*byte) << (8 * index))
        });
    ((value >> shift) & mask, mask, span)
}

/// Entries of one width at consecutive bit offsets of a little-endian
/// buffer: dense elements, or codes packed into `u32` words.
#[derive(Clone, Debug)]
struct Stream {
    steps: Steps,
    entry_bits: u32,
    chunk_bytes: usize,
}

/// One plane of a packet form.
#[derive(Clone, Debug)]
struct Plane {
    steps: Steps,
    /// First bit of the plane within a packet of the `packet` layout.
    offset_bits: usize,
    entry_bits: u32,
    /// The runs of one entry in the tensor's layout.
    runs: Vec<Run>,
    /// Logical values sharing one group of `fields` entries.
    group: u64,
    fields: u64,
    entries_per_packet: u64,
}

impl Plane {
    /// Entries read by the first `values` logical values of a packet.
    fn entries(&self, values: u64) -> u64 {
        (values - 1) / self.group * self.fields + self.fields
    }
}

/// Where the packet-form planes of a packed tensor live.
#[derive(Clone, Debug)]
enum Placement {
    /// Chunk `c` is the packet of cell `c`.
    Packet,
    /// Chunk `t` is row tile `t`.
    Rows {
        layout: &'static PackedRowLayout,
        geometry: RowGeometry,
        extents: Vec<u64>,
        /// Logical and stored rows of one matrix (the last leading extent,
        /// and that extent padded to whole tiles).
        matrix_rows: u64,
        stored_matrix_rows: u64,
    },
}

#[derive(Clone, Debug)]
struct Packed {
    planes: Vec<Plane>,
    placement: Placement,
    /// Logical values of one packet, and of one row.
    group: u64,
    extent: u64,
    packets_per_row: u64,
    chunk_bytes: usize,
    chunks: u64,
}

#[derive(Clone, Debug)]
enum Storage {
    Stream(Stream),
    Packed(Packed),
}

/// What a buffer stores, as the lattices its stored values step on.
#[derive(Clone, Debug)]
pub struct Lattice(Storage);

impl Lattice {
    /// Dense storage of the floating dtype `dtype`, little-endian.
    pub fn dense(dtype: DType) -> Result<Lattice, LatticeError> {
        let steps = Steps::float(dtype).ok_or(LatticeError::NotFloating(dtype))?;
        Ok(Lattice(Storage::Stream(Stream {
            steps,
            entry_bits: dtype.bytes() * 8,
            chunk_bytes: dtype.bytes() as usize,
        })))
    }

    /// Codes of `bits` bits packed into little-endian `u32` words, meaningful
    /// in the inclusive range `codes`.
    pub fn codes(bits: u32, codes: RangeInclusive<u32>) -> Result<Lattice, LatticeError> {
        if bits == 0 || !32u32.is_multiple_of(bits) {
            return Err(LatticeError::CodeWidth(bits));
        }
        if codes.is_empty() || *codes.end() > u32::MAX >> (32 - bits) {
            return Err(LatticeError::CodeRange);
        }
        Ok(Lattice(Storage::Stream(Stream {
            steps: Steps::Range {
                min: *codes.start(),
                max: *codes.end(),
            },
            entry_bits: bits,
            chunk_bytes: 4,
        })))
    }

    /// The canonical bytes of a tensor with `extents` in the packed storage
    /// `representation`, in the `packet` layout or any row layout.
    pub fn packed(
        representation: RepresentationId,
        extents: &[u64],
    ) -> Result<Lattice, LatticeError> {
        let info = registry::representation_info(representation);
        let (packet, rows): (
            &'static PackedPacketLayout,
            Option<&'static PackedRowLayout>,
        ) = match &info.kind {
            RepresentationKind::Packed(packet) => (packet, None),
            RepresentationKind::PackedRows(rows) => (&rows.packet, Some(rows)),
            RepresentationKind::Dense(_) | RepresentationKind::External(_) => {
                return Err(LatticeError::NotPacked(info.name));
            }
        };
        let bytes = registry::canonical_bytes(representation, extents)
            .filter(|bytes| *bytes != 0 && usize::try_from(*bytes).is_ok())
            .ok_or(LatticeError::Geometry)?;
        let (extent, leading) = extents
            .split_last()
            .expect("a packed tensor with a canonical byte count has a packing axis");
        let group = u64::from(packet.group);
        let planes =
            packet
                .planes
                .iter()
                .enumerate()
                .map(|(ordinal, plane)| {
                    let steps = Steps::of(&plane.encoding).ok_or(LatticeError::Plane {
                        representation: info.representation,
                        plane: plane.name,
                    })?;
                    let whole = Run {
                        first: 0,
                        bits: plane.entry_bits,
                    };
                    // A row layout splits the codes of packet plane 0 across its
                    // code planes, each holding a run of every code's bits, unless
                    // it keeps vector-code entries whole (`grids`); every other
                    // packet plane keeps its bytes together (`RowPlaneContent`).
                    // The `packet` layout keeps every plane's bytes together.
                    let sliced = rows.is_some_and(|layout| {
                        layout.planes.iter().any(|row_plane| {
                            matches!(row_plane.content, RowPlaneContent::Codes { .. })
                        })
                    });
                    let runs = match rows {
                        Some(layout) if ordinal == 0 && sliced => {
                            let mut runs: Vec<Run> = layout
                                .planes
                                .iter()
                                .filter_map(|row_plane| match row_plane.content {
                                    RowPlaneContent::Codes { shift, bits } => {
                                        Some(Run { first: shift, bits })
                                    }
                                    RowPlaneContent::Grids { .. }
                                    | RowPlaneContent::Groups { .. } => None,
                                })
                                .collect();
                            runs.sort_by_key(|run| run.first);
                            let covered = runs.iter().try_fold(0, |covered, run| {
                                (run.first == covered).then_some(covered + run.bits)
                            });
                            assert_eq!(
                                covered,
                                Some(plane.entry_bits),
                                "every code bit has one row code plane"
                            );
                            runs
                        }
                        Some(_) | None => vec![whole],
                    };
                    let mut plane = Plane {
                        steps,
                        offset_bits: plane.offset as usize * 8,
                        entry_bits: plane.entry_bits,
                        runs,
                        group: u64::from(plane.group),
                        fields: u64::from(plane.fields),
                        entries_per_packet: 0,
                    };
                    plane.entries_per_packet = plane.entries(group);
                    Ok(plane)
                })
                .collect::<Result<Vec<_>, LatticeError>>()?;
        let (placement, chunk_bytes) = match rows {
            None => (Placement::Packet, u64::from(packet.packet_size)),
            Some(layout) => {
                let geometry = layout
                    .geometry(*extent)
                    .expect("a row layout with a canonical byte count has a row geometry");
                let units = layout
                    .storage_units(extents)
                    .expect("a row layout with a canonical byte count has storage units");
                let chunk_bytes = layout.tile_rows() * geometry.stride;
                let (matrix_rows, stored_matrix_rows) = match leading.last() {
                    Some(rows) => (*rows, units[units.len() - 2]),
                    None => (1, 1),
                };
                let placement = Placement::Rows {
                    layout,
                    geometry,
                    extents: extents.to_vec(),
                    matrix_rows,
                    stored_matrix_rows,
                };
                (placement, chunk_bytes)
            }
        };
        Ok(Lattice(Storage::Packed(Packed {
            planes,
            placement,
            group,
            extent: *extent,
            packets_per_row: packet.packet_extent(*extent),
            chunk_bytes: chunk_bytes as usize,
            chunks: bytes / chunk_bytes,
        })))
    }

    /// Bytes of one chunk. Buffers passed to [`Lattice::perturb`] hold whole
    /// chunks.
    pub fn chunk_bytes(&self) -> usize {
        match &self.0 {
            Storage::Stream(stream) => stream.chunk_bytes,
            Storage::Packed(packed) => packed.chunk_bytes,
        }
    }

    /// Step the stored values of `bytes`, the whole chunks starting at chunk
    /// `first_chunk` of the buffer, and count the realised moves.
    ///
    /// The result holds one tally for dense storage and for a code plane, and
    /// one tally per packet-form plane, in the registry's plane order, for a
    /// packed representation. Tallies of the chunks of a buffer add.
    ///
    /// A byte count that is not whole chunks, or chunks beyond a packed
    /// tensor's canonical bytes, contradict the description and panic.
    pub fn perturb(&self, dither: &Dither, first_chunk: u64, bytes: &mut [u8]) -> Vec<Tally> {
        let chunk_bytes = self.chunk_bytes();
        assert!(
            bytes.len().is_multiple_of(chunk_bytes),
            "{} bytes are not whole {chunk_bytes}-byte chunks",
            bytes.len()
        );
        let chunks = (bytes.len() / chunk_bytes) as u64;
        match &self.0 {
            Storage::Stream(stream) => {
                let key = dither.stream(0);
                let mut tally = Tally::default();
                let width = stream.entry_bits as usize;
                let first = first_chunk * (chunk_bytes * 8 / width) as u64;
                let runs = [Run {
                    first: 0,
                    bits: stream.entry_bits,
                }];
                tally.values = (bytes.len() * 8 / width) as u64;
                for entry in 0..tally.values {
                    if let Some(direction) = dither.draw(key, first + entry) {
                        stream.steps.apply(
                            bytes,
                            &runs,
                            |bit| entry as usize * width + bit as usize,
                            direction,
                            &mut tally,
                        );
                    }
                }
                vec![tally]
            }
            Storage::Packed(packed) => {
                assert!(
                    first_chunk
                        .checked_add(chunks)
                        .is_some_and(|end| end <= packed.chunks),
                    "chunks {first_chunk}..+{chunks} lie outside the tensor's {} chunks",
                    packed.chunks
                );
                packed.perturb(dither, first_chunk, chunks, bytes)
            }
        }
    }
}

impl Packed {
    fn perturb(
        &self,
        dither: &Dither,
        first_chunk: u64,
        chunks: u64,
        bytes: &mut [u8],
    ) -> Vec<Tally> {
        let streams: Vec<u64> = (0..self.planes.len())
            .map(|ordinal| dither.stream(ordinal))
            .collect();
        let mut tallies = vec![Tally::default(); self.planes.len()];
        match &self.placement {
            Placement::Packet => {
                for chunk in 0..chunks {
                    let origin = chunk as usize * self.chunk_bytes * 8;
                    self.step_cell(
                        dither,
                        &streams,
                        first_chunk + chunk,
                        bytes,
                        &mut tallies,
                        |plane, bit| origin + self.planes[plane].offset_bits + bit as usize,
                    );
                }
            }
            Placement::Rows {
                layout,
                geometry,
                extents,
                matrix_rows,
                stored_matrix_rows,
            } => {
                let tile = layout.tile_rows();
                let origin = first_chunk * tile * geometry.stride * 8;
                for stored in first_chunk * tile..(first_chunk + chunks) * tile {
                    let within = stored % stored_matrix_rows;
                    if within >= *matrix_rows {
                        continue;
                    }
                    let row = stored / stored_matrix_rows * matrix_rows + within;
                    assert_eq!(
                        layout.stored_row(extents, row),
                        stored,
                        "row layout stores its rows in storage-unit order"
                    );
                    for packet in 0..self.packets_per_row {
                        self.step_cell(
                            dither,
                            &streams,
                            row * self.packets_per_row + packet,
                            bytes,
                            &mut tallies,
                            |plane, bit| {
                                (layout.packet_bit(geometry, stored, packet, plane, bit) - origin)
                                    as usize
                            },
                        );
                    }
                }
            }
        }
        tallies
    }

    /// Step the stored entries of cell `cell` (packet `cell % packets_per_row`
    /// of logical row `cell / packets_per_row`). `bit_of(plane, bit)` is the
    /// buffer bit holding bit `bit` of packet-form plane `plane` of the cell.
    fn step_cell(
        &self,
        dither: &Dither,
        streams: &[u64],
        cell: u64,
        bytes: &mut [u8],
        tallies: &mut [Tally],
        bit_of: impl Fn(usize, u32) -> usize,
    ) {
        let packet = cell % self.packets_per_row;
        let values = (self.extent - packet * self.group).min(self.group);
        for (ordinal, plane) in self.planes.iter().enumerate() {
            let entries = plane.entries(values);
            let first = cell * plane.entries_per_packet;
            for entry in 0..entries {
                if let Some(direction) = dither.draw(streams[ordinal], first + entry) {
                    plane.steps.apply(
                        bytes,
                        &plane.runs,
                        |bit| bit_of(ordinal, entry as u32 * plane.entry_bits + bit),
                        direction,
                        &mut tallies[ordinal],
                    );
                }
            }
            tallies[ordinal].values += entries;
        }
    }
}
