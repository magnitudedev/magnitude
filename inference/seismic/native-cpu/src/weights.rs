//! Weight elements a CPU weight operand can be bound to, their row
//! geometry, and the bodies of their components.
//!
//! A weight row of `k` logical values is read in packets of 32 consecutive
//! values. Dense rows (`f32`, `bf16`, `f16`) hold the values contiguously; a
//! partial final packet decodes its missing values as zero. Packed rows are
//! the registry's row layouts (the CPU's `rows16` and `rows8`, and the
//! devices' `rows32` and `mma16`, read where they lie, [`Placement`]) with
//! the registry's geometry ([`RowGeometry::of`]), decoded by the components
//! the registry's
//! decode axes name (`registry::decode_axes`, instantiated by build.rs): a
//! code component ([`Planes`], [`Grid`]) gives a packet's exact integer
//! values, and a coefficient component ([`Direct`], [`Hier`]) the scale (and
//! bias) of each of its coefficient groups. A value is `scale * value + bias`
//! (fused) or `scale * value`, rounded once to `f32` as the registry's
//! decode recipe rounds it.
//!
//! The dot bodies accumulate `w[i] * x[i]` (fused) into lane `i % LANES` in
//! index order and combine the lanes in the fixed tree of `reduce`, so a
//! result depends only on the values, never on the tier or the row block.
//! The quantized dot bodies sum exact integer products and combine them in
//! `f32` in the fixed order of their coefficient component, so they too
//! depend only on the values: the instructions forming the integer sums are
//! free to differ by tier.

use crate::element::{bf16_to_f32, f16_to_f32, Dense};
use crate::isa::Isa;
use crate::quant::{Q8Block, Q8Tile, Q8_BLOCK};
use crate::reduce::{combine, LANES};
use magnitude_ggml::tables;
use seismic_lang::registry::{
    self, GridSigns, Layout, LocalBias, MinimumSign, RepresentationInfo, RepresentationKind,
    RowPlaneContent, RowPlaneRole,
};
use std::marker::PhantomData;

mod tile;

/// Values of one packet.
pub const PACKET: usize = 32;

/// How a row layout places its tile's rows (`registry::Layout`), as the
/// components address them (`PackedRowLayout::code_bit`, `share_byte`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Placement {
    /// Dense rows and `rows16`: each row's planes within the row, a
    /// storage group's share of a plane its bytes together.
    Rows,
    /// `rows8`: a tile's plane a span per storage group, a code plane's in
    /// panel order, a group plane's record by record, a grid plane's its
    /// rows' shares in turn (`registry::Layout::Rows8`).
    Rows8,
    /// `rows32`: a tile's planes one after another, a group plane its rows'
    /// shares row by row, a code plane each packet's 32 columns of its rows
    /// together (`[packet][row][packet bytes]`).
    Rows32,
    /// `mma16`: group planes as `rows16`'s; a tile's code plane spread over
    /// its rows' code planes, each k-block of 64 columns of the tile its
    /// GEMV lanes (codes: `[k-block][lane 4g + t][4 steps of 16 columns]`,
    /// row `g` and `g + 8` sharing each lane) or rows (grids: `[k-block][row]`).
    Mma16,
    /// `mma16` of a carried grid representation: its one `records` plane, a
    /// tile's bytes whole (`PackedRowLayout::record_byte`), each storage
    /// group's `entries` bytes of its rows' grid entries in GEMV lane order
    /// (`PackedRowLayout::record_entry`), then every group's factors.
    Records { entries: usize },
}

impl Placement {
    /// Rows of one tile.
    pub const fn tile(self) -> usize {
        match self {
            Self::Rows => 1,
            Self::Rows8 => 8,
            Self::Rows32 => 32,
            Self::Mma16 | Self::Records { .. } => 16,
        }
    }

    /// Rows of the dot block a projection takes at a time
    /// (`components::ROW_BLOCKS`): an `mma16` tile, whose rows `g` and
    /// `g + 8` share their code bytes (read once for both); eight rows of
    /// every other layout, whose rows share none (a rows8 tile's eight are
    /// its dot block too).
    pub const fn row_block(self) -> usize {
        match self {
            Self::Mma16 => 16,
            Self::Rows | Self::Rows8 | Self::Rows32 | Self::Records { .. } => 8,
        }
    }
}

/// The stack alignment every frame has (the x86-64 and AArch64 ABIs').
const STACK_ALIGNMENT: usize = 16;

/// A packet buffer the vector bodies store and reload whole, starting on a
/// `VECTOR`-byte boundary (their tier's widest vector, `Tier::vector_bytes`):
/// wherever a caller's frame puts it, no vector access straddles a cache
/// line, which stalls store-to-load forwarding. Its `LENGTH` values
/// (`VECTOR`'s room, a parameter of its own because stable array lengths
/// take no expression of one) keep only the ABI's stack alignment, with
/// room to start at the next vector boundary, so no frame is realigned
/// (which would cost the body its frame-pointer register).
#[derive(Clone, Copy)]
#[repr(C, align(16))]
pub struct PacketBuffer<const VECTOR: usize, const LENGTH: usize>([f32; LENGTH]);

impl<const VECTOR: usize, const LENGTH: usize> PacketBuffer<VECTOR, LENGTH> {
    const ZERO: Self = {
        assert!(
            std::mem::align_of::<Self>() == STACK_ALIGNMENT
                && LENGTH == PACKET + (VECTOR - STACK_ALIGNMENT) / 4,
            "a packet buffer holds a packet from its first vector boundary"
        );
        Self([0.0; LENGTH])
    };

    /// The packet, from the first vector boundary of the storage.
    #[inline(always)]
    fn packet(&mut self) -> &mut [f32; PACKET] {
        let skip = if VECTOR == STACK_ALIGNMENT {
            0
        } else {
            self.0.as_ptr().align_offset(VECTOR)
        };
        (&mut self.0[skip..skip + PACKET])
            .try_into()
            .expect("a packet")
    }
}

/// A tier as the bodies take it: its token type (`isa::Isa`), whose
/// quantized-dot mode and packet buffer they compile for.
pub trait Vectors: Isa {
    /// The tier's packet buffer.
    type Buffer: Copy;
    const ZERO: Self::Buffer;
    /// The packet of `buffer`.
    fn packet(buffer: &mut Self::Buffer) -> &mut [f32; PACKET];
}

/// `Vectors` for every tier token of this architecture, one per tier of
/// `seismic_cpu_tiers::tiers!`.
macro_rules! vectors {
    ($($token:ident $name:ident $arch:tt [$($feature:tt),*];)*) => {$(
        #[cfg(target_arch = $arch)]
        impl Vectors for crate::isa::$token {
            type Buffer = PacketBuffer<
                { <crate::isa::$token as Isa>::TIER.vector_bytes() },
                { PACKET + (<crate::isa::$token as Isa>::TIER.vector_bytes() - STACK_ALIGNMENT) / 4 },
            >;
            const ZERO: Self::Buffer = PacketBuffer::ZERO;

            #[inline(always)]
            fn packet(buffer: &mut Self::Buffer) -> &mut [f32; PACKET] {
                buffer.packet()
            }
        }
    )*};
}

seismic_cpu_tiers::tiers!(vectors);

/// Byte geometry of the rows of one weight operand.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RowGeometry {
    /// Bytes between the starts of adjacent rows.
    pub stride: usize,
    /// Per plane role (`RowPlaneRole::ALL` order), its offset within a row
    /// and its bytes per storage group; zero for a role the element lacks.
    pub offsets: [usize; 4],
    pub groups: [usize; 4],
    /// Per plane role, its payload bytes of one row (`bytes_per_row`).
    pub row_bytes: [usize; 4],
    /// Packets (of 32 values) of one storage group.
    pub group_packets: usize,
    /// Storage groups of every row before the operand's first value (a
    /// window of its columns, `Weights::columns`).
    pub first_group: usize,
    pub placement: Placement,
    /// Whether the code plane holds grid entries (`RowPlaneContent::Grids`).
    pub grids: bool,
    /// Row addressing of the bound tensor. Zero base is used for unbound geometry.
    pub base: usize,
    pub matrix_rows: usize,
    /// Per code role of an `mma16` layout, the bytes of one unit of a
    /// tile's `V`, the units of one row's payload and their division by
    /// multiplication (`Mma16Plane`): derived once, not per row.
    pub mma16_units: [(usize, usize, u64); 2],
}

impl RowGeometry {
    /// The geometry of rows of `k` values of weight element `element` (dense,
    /// or packed in any row layout);
    /// `dense_stride` is the bound tensor's row stride in bytes, which only
    /// dense rows use. Packed rows follow the registry's row layout
    /// (`PackedRowLayout::geometry`), computed without allocating.
    pub fn of(element: &RepresentationInfo, k: usize, dense_stride: usize) -> Self {
        let mut geometry = Self {
            stride: dense_stride,
            offsets: [0; 4],
            groups: [0; 4],
            row_bytes: [0; 4],
            group_packets: 1,
            first_group: 0,
            placement: Placement::Rows,
            grids: false,
            base: 0,
            matrix_rows: 0,
            mma16_units: [(0, 0, 0); 2],
        };
        match &element.kind {
            RepresentationKind::Dense(_) => {}
            RepresentationKind::PackedRows(rows) => {
                let k = k as u64;
                geometry.stride = rows.row_stride_bytes(k).expect("a weight row's stride") as usize;
                for (plane, row_plane) in rows.planes.iter().enumerate() {
                    let role = row_plane.role() as usize;
                    geometry.offsets[role] =
                        rows.plane_row_offset(plane, k)
                            .expect("a weight plane's offset") as usize;
                    geometry.groups[role] = row_plane.bytes_per_group as usize;
                    geometry.row_bytes[role] = rows
                        .plane_bytes_per_row(plane, k)
                        .expect("a weight plane's bytes") as usize;
                    if matches!(row_plane.content, RowPlaneContent::Grids { .. }) {
                        geometry.grids = true;
                    }
                }
                geometry.group_packets = rows.group() as usize / PACKET;
                let records = rows
                    .planes
                    .iter()
                    .any(|plane| matches!(plane.content, RowPlaneContent::Records { .. }));
                geometry.placement = match element.layout {
                    Layout::Rows16 => Placement::Rows,
                    Layout::Rows8 => Placement::Rows8,
                    Layout::Rows32 => Placement::Rows32,
                    Layout::Mma16 if records => {
                        Placement::Records { entries: rows.record_entry_bytes() as usize }
                    }
                    Layout::Mma16 => Placement::Mma16,
                    Layout::Packet => unreachable!("a row layout is never `packet`"),
                };
            }
            RepresentationKind::Packed(_) | RepresentationKind::External(_) => {
                panic!("`{}` is no CPU weight element", element.name)
            }
        }
        if geometry.placement == Placement::Mma16 {
            for role in RowPlaneRole::CODES {
                let role = role as usize;
                // A lane holds a packet's bytes of one k-block (four steps
                // of 16 columns, two rows); a grid row's k-block is two
                // packets.
                let packet = geometry.packet_bytes(role);
                let unit = if geometry.grids { 2 * packet } else { packet };
                let units = (geometry.row_bytes[role] / unit.max(1)).max(1);
                geometry.mma16_units[role] = (unit, units, (1u64 << 40) / units as u64 + 1);
            }
        }
        geometry
    }

    pub(crate) fn with_base(mut self, base: *const u8, matrix_rows: usize) -> Self {
        self.base = base as usize;
        self.matrix_rows = matrix_rows;
        self
    }

    /// Bytes of one packet's codes of one row in code plane `role`.
    #[inline(always)]
    fn packet_bytes(&self, role: usize) -> usize {
        self.groups[role] / self.group_packets
    }

    /// The tile base and the row within the tile of the row at `row`
    /// (every matrix's rows padded to whole tiles).
    #[inline(always)]
    fn tile_row(&self, row: *const u8) -> (*const u8, usize) {
        let tile = self.placement.tile();
        let index = (row as usize - self.base) / self.stride;
        let n = self.matrix_rows;
        let stored = index / n * n.div_ceil(tile) * tile + index % n;
        (
            (self.base + stored / tile * self.stride * tile) as *const u8,
            stored % tile,
        )
    }

    /// `R` consecutive rows from `first`, each resolved as [`Self::row`]
    /// resolves it, with one division for all: a row's place in its tile
    /// advances by one, past each matrix's padding to the next.
    #[inline(always)]
    pub(crate) fn rows<const R: usize>(&self, first: *const u8) -> [Row; R] {
        if self.placement == Placement::Rows {
            return std::array::from_fn(|r| self.row_in(first.wrapping_add(r * self.stride), (first.wrapping_add(r * self.stride), 0)));
        }
        let tile = self.placement.tile();
        let n = self.matrix_rows;
        let index = (first as usize - self.base) / self.stride;
        let (mut matrix, mut within) = (index / n, index % n);
        let padded = n.div_ceil(tile) * tile;
        std::array::from_fn(|r| {
            // Tiles are powers of two: a mask, no division a row.
            let stored = matrix * padded + within;
            let lane = stored & (tile - 1);
            let place = ((self.base + (stored - lane) * self.stride) as *const u8, lane);
            let row = self.row_in(first.wrapping_add(r * self.stride), place);
            within += 1;
            if within == n {
                (matrix, within) = (matrix + 1, 0);
            }
            row
        })
    }

    /// The row at `row`, resolved once for the component call that reads
    /// it.
    #[inline(always)]
    pub(crate) fn row(&self, row: *const u8) -> Row {
        let place = match self.placement {
            Placement::Rows => (row, 0),
            _ => self.tile_row(row),
        };
        self.row_in(row, place)
    }

    /// The row at `row`, its tile's first byte and its place in it `(tile,
    /// r)`.
    #[inline(always)]
    fn row_in(&self, row: *const u8, (tile, r): (*const u8, usize)) -> Row {
        let mut planes = [row; 4];
        let mut steps = self.groups;
        let mut packet_steps = [0; 2];
        for role in RowPlaneRole::CODES {
            packet_steps[role as usize] = self.packet_bytes(role as usize);
        }
        for plane in 0..4 {
            let (offset, bytes) = (self.offsets[plane], self.groups[plane]);
            let code = plane < 2;
            planes[plane] = match self.placement {
                Placement::Rows => row.wrapping_add(offset),
                // A panel unit holds row `r`'s codes 4 bytes in, 32 bytes a
                // unit; a grid span its rows' shares in turn; a group span
                // its records' rows in turn, read through `At::field`.
                Placement::Rows8 => {
                    steps[plane] = bytes * 8;
                    let within = match (code, self.grids) {
                        (true, false) => {
                            packet_steps[plane] = self.packet_bytes(plane) * 8;
                            4 * r
                        }
                        (true, true) => r * bytes,
                        (false, _) => 0,
                    };
                    tile.wrapping_add(offset * 8 + within)
                }
                Placement::Rows32 if code => {
                    let packet = self.packet_bytes(plane);
                    steps[plane] = bytes * 32;
                    packet_steps[plane] = packet * 32;
                    tile.wrapping_add(offset * 32 + r * packet)
                }
                Placement::Rows32 => tile.wrapping_add(offset * 32 + r * self.row_bytes[plane]),
                Placement::Mma16 if code => std::ptr::null(),
                Placement::Mma16 => tile.wrapping_add(r * self.stride + offset),
                // The one plane's storage groups: their entries (at
                // `record_entry` of each), and, past every stored group's
                // entries, their factors, the row's at `record_factor`.
                Placement::Records { entries } => {
                    let record = self.groups[RowPlaneRole::Codes as usize];
                    let factors = registry::MMA_TILE_ROWS as usize * record - entries;
                    match RowPlaneRole::ALL[plane] {
                        RowPlaneRole::Codes => {
                            steps[plane] = entries;
                            tile
                        }
                        RowPlaneRole::Supers => {
                            steps[plane] = factors;
                            let stored = self.row_bytes[RowPlaneRole::Codes as usize] / record;
                            let width = factors / registry::MMA_TILE_ROWS as usize;
                            let place = registry::PackedRowLayout::record_factor(r as u64) as usize;
                            tile.wrapping_add(stored * entries + place * width)
                        }
                        RowPlaneRole::High | RowPlaneRole::Scales => std::ptr::null(),
                    }
                }
            };
            planes[plane] = planes[plane].wrapping_add(self.first_group * steps[plane]);
        }
        Row {
            start: row,
            planes,
            steps,
            packet_steps,
            tile: (self.placement == Placement::Rows8).then_some((tile as usize, r)),
            panel: self.placement == Placement::Rows8 && !self.grids,
            mma16: (self.placement == Placement::Mma16).then(|| Mma16Row::of(self, tile, r)),
            record: matches!(self.placement, Placement::Records { .. }).then_some(r),
        }
    }
}

/// A row's code planes of an `mma16` tile: each plane's bytes of the tile,
/// `V`, are its 16 rows' plane payloads one after another
/// (`PackedRowLayout::code_bit`), so `V`'s units (a lane's or a row's bytes
/// of one k-block) lie in the rows' planes in turn.
#[derive(Clone, Copy, Debug)]
struct Mma16Row {
    /// The row within its tile.
    row: usize,
    /// The view's first packet of a row, and the packets of a storage group.
    first_packet: usize,
    group_packets: usize,
    planes: [Mma16Plane; 2],
}

/// One code plane of an `mma16` tile (`Mma16Row`).
#[derive(Clone, Copy, Debug)]
struct Mma16Plane {
    /// The plane's first byte in the tile's first row.
    start: *const u8,
    stride: usize,
    /// Bytes of one unit of `V`: a lane's bytes of one k-block (codes), or
    /// a row's (grids); and the units of one row's payload, with its
    /// division by multiplication (`units * magic` within `2^40` of a
    /// multiple of `2^40`, exact for every unit of a tile).
    unit: usize,
    units: usize,
    magic: u64,
    /// Whether it holds grid entries or their further bits (`[k-block][row]`,
    /// a grid high plane placed as its grids plane), else codes by lanes.
    grid: bool,
}

impl Mma16Row {
    fn of(geometry: &RowGeometry, tile: *const u8, row: usize) -> Self {
        let plane = |role: usize| {
            let (unit, units, magic) = geometry.mma16_units[role];
            Mma16Plane {
                start: tile.wrapping_add(geometry.offsets[role]),
                stride: geometry.stride,
                unit,
                units,
                magic,
                grid: geometry.grids,
            }
        };
        Self {
            row,
            first_packet: geometry.first_group * geometry.group_packets,
            group_packets: geometry.group_packets,
            planes: [plane(0), plane(1)],
        }
    }
}

impl Mma16Plane {
    /// The first byte of unit `unit` of the tile's `V`.
    #[inline(always)]
    fn unit(&self, unit: usize) -> *const u8 {
        self.at(self.place(unit))
    }

    /// The row of the tile and the unit within its payload holding unit
    /// `unit` of `V`: the same in every code plane of the tile with as many
    /// units a row (a codes plane and its high plane).
    #[inline(always)]
    fn place(&self, unit: usize) -> (usize, usize) {
        let row = ((unit as u64 * self.magic) >> 40) as usize;
        (row, unit - row * self.units)
    }

    /// The first bytes of lane quads `4g ..4g + 4` of the k-block whose
    /// lane 0 is unit `first` of `V` (each quad together: `Row::lanes_whole`),
    /// with one division where the k-block lies in one row's payload. The
    /// x86 tile path's alone.
    #[cfg(target_arch = "x86_64")]
    #[inline(always)]
    fn quads(&self, first: usize) -> [*const u8; 8] {
        let (row, within) = self.place(first);
        if within + 32 <= self.units {
            let base = self.at((row, within));
            std::array::from_fn(|g| base.wrapping_add(4 * g * self.unit))
        } else {
            std::array::from_fn(|g| self.unit(first + 4 * g))
        }
    }

    /// The first byte of unit `within` of the tile's row `row`'s payload.
    #[inline(always)]
    fn at(&self, (row, within): (usize, usize)) -> *const u8 {
        self.start.wrapping_add(row * self.stride + within * self.unit)
    }
}

/// One row of a weight operand, resolved once per component call: where
/// each plane's storage groups lie.
#[derive(Clone, Copy, Debug)]
pub struct Row {
    /// The row's first byte: a dense row's values.
    start: *const u8,
    /// Per plane role (`RowPlaneRole` order), the first byte of storage
    /// group 0 and the bytes from one storage group to the next, a group's
    /// share of a plane at its first byte (but an `mma16` code plane's,
    /// `mma16`).
    planes: [*const u8; 4],
    steps: [usize; 4],
    /// Per code role, the bytes from one packet's codes to the next within
    /// a storage group's share.
    packet_steps: [usize; 2],
    /// The rows8 tile base and lane.
    tile: Option<(usize, usize)>,
    /// Whether the code planes are rows8 panels (`CodeBytes::Panel`).
    panel: bool,
    /// An `mma16` row's code planes.
    mma16: Option<Mma16Row>,
    /// A `records` row's place in its tile.
    record: Option<usize>,
}

impl Row {
    /// Whether the first `live` of eight rows are lanes `0..live` of one
    /// rows8 tile of panels, in order: the tile's spans then hold their
    /// codes and coefficients (its other lanes whatever they hold).
    #[inline(always)]
    fn panel_tile(rows: &[Row; 8], live: usize) -> bool {
        rows[0].panel
            && rows[0].tile.is_some_and(|(base, lane)| {
                lane == 0 && (1..live).all(|r| rows[r].tile == Some((base, r)))
            })
    }

    /// The row `bytes` on, in a layout whose rows (and an `mma16` tile's
    /// code planes) lie at a constant pitch: the same place in a tile
    /// `bytes` further. The x86 tile path's alone.
    #[cfg(target_arch = "x86_64")]
    #[inline(always)]
    fn shifted(&self, bytes: usize) -> Row {
        let mut row = *self;
        row.start = self.start.wrapping_add(bytes);
        for role in [RowPlaneRole::Scales, RowPlaneRole::Supers] {
            let role = role as usize;
            row.planes[role] = self.planes[role].wrapping_add(bytes);
        }
        if let Some(mma16) = row.mma16.as_mut() {
            for plane in &mut mma16.planes {
                plane.start = plane.start.wrapping_add(bytes);
            }
        }
        row
    }

    /// Whether every four lanes `4g ..4g + 4` of a k-block of this `mma16`
    /// row's code planes lie together in one row's payload: a row's payload
    /// being whole lane quads. The x86 tile path's alone.
    #[cfg(target_arch = "x86_64")]
    #[inline(always)]
    fn lanes_whole(&self) -> bool {
        self.mma16.as_ref().is_some_and(|mma16| !mma16.planes[0].grid && mma16.planes[0].units % 4 == 0)
    }

    /// Whether sixteen rows are rows 0..16 of one `mma16` tile whose code
    /// plane holds lanes, in order: rows `g` and `g + 8` then share their
    /// code bytes (`Codes::neon_partners`).
    #[inline(always)]
    fn partners<const R: usize>(rows: &[Row; R]) -> bool {
        let tile = |row: &Row| row.mma16.as_ref().filter(|mma16| !mma16.planes[0].grid).map(|mma16| (mma16.planes[0].start, mma16.row));
        R == 16
            && tile(&rows[0]).is_some_and(|(start, row)| {
                row == 0 && (1..R).all(|r| tile(&rows[r]) == Some((start, r)))
            })
    }

    /// The first byte of storage group `index` of every plane.
    #[inline(always)]
    fn group(&self, index: usize) -> [*const u8; 4] {
        let mut planes = self.planes;
        for plane in 0..4 {
            planes[plane] = planes[plane].wrapping_add(index * self.steps[plane]);
        }
        planes
    }
}

/// One weight element's component bodies.
pub trait Format: Copy + Send + Sync + 'static {
    /// The 32 values of packet `p` of `row`, on tier `V`.
    ///
    /// # Safety
    /// `row` is a row of at least `32 * p + 1` values.
    unsafe fn decode<V: Vectors>(row: &Row, p: usize, k: usize, out: &mut [f32; PACKET]);

    /// `lanes` plus the products of whole packet `p` of `row` and `x`, value
    /// `i` fused into lane `i % LANES` in index order ([`accumulate`]):
    /// its decode into `packet`, then the products; an element whose values
    /// widen in the tier's vectors feeds them to the products directly.
    ///
    /// # Safety
    /// `row` is a row of at least `32 (p + 1)` values.
    #[inline(always)]
    unsafe fn dot_packet<V: Vectors>(
        row: &Row,
        p: usize,
        k: usize,
        lanes: &mut [f32; LANES],
        x: &[f32; PACKET],
        packet: &mut [f32; PACKET],
    ) {
        unsafe { Self::decode::<V>(row, p, k, packet) };
        unsafe { accumulate::<V>(lanes, packet, x) };
    }

    /// Values `256 b .. 256 b + 256` (below `k`) of each of the `R` rows
    /// `rows` dotted with each of the `M` quantized activation blocks `x`: per row and activation block, the
    /// same exact integer sums and the same `f32` combine for every `R` and
    /// `M`.
    ///
    /// # Safety
    /// Each row is a row of `k` values, and `256 b < k`.
    unsafe fn dot_q8<const R: usize, const M: usize, V: Vectors>(
        rows: &[Row; R],
        b: usize,
        k: usize,
        x: [&Q8Block; M],
    ) -> [[f32; M]; R];

    /// [`dot_q8_rows`] of `out.len()` rows from `first` against one
    /// activation row, every 16 a tile,
    /// written into `out` where the format has a whole-tile form for them on
    /// tier `V` (`mma16` tiles of nibble-lane codes, `rows16` rows of
    /// unsigned byte values): the same sums bit for
    /// bit. False, with `out` untouched, where it has none.
    ///
    /// # Safety
    /// As [`dot_q8_rows`], for `out.len()` rows.
    #[inline(always)]
    unsafe fn dot_q8_tiles<V: Vectors>(first: *const u8, geometry: &RowGeometry, x: &[Q8Block], k: usize, out: &mut [f32]) -> bool {
        let _ = (first, geometry, x, k, out);
        false
    }

    /// The quantized dot of `row` (of `k` values) and `x`, from
    /// [`Coefficients::reference`] block by block: the tests' reference.
    ///
    /// # Safety
    /// `row` is a row of the format of `k` values.
    #[cfg(test)]
    unsafe fn reference_q8(row: &Row, k: usize, x: &[Q8Block]) -> f32 {
        let _ = (row, k, x);
        unreachable!("only packed integer codes have a quantized reference")
    }

    /// Whether the format has the tile form ([`Format::tile_q8`]): packed
    /// integer codes.
    const TILED: bool = false;

    /// `out[a n + r] = (row first + r) · activation a` for `n` rows from
    /// `first` against the `m` activation rows of `x` (`quant::tiles(m,
    /// blocks)` tiles): each weight block decoded once for every activation
    /// row, each output the sums [`dot_q8_rows`] gives the row alone, bit
    /// for bit (`tile`).
    ///
    /// # Safety
    /// `first` addresses `n` rows with `geometry` and `k` values; the format
    /// is [`Format::TILED`].
    unsafe fn tile_q8<V: Vectors, const R: usize>(first: *const u8, geometry: &RowGeometry, n: usize, x: &[Q8Tile<R>], m: usize, k: usize, out: &mut [f32]) {
        let _ = (first, geometry, n, x, m, k, out);
        unreachable!("only packed integer codes take the tile form")
    }
}

/// Packets `0..count` of a block (`count` at most 8) as a loop bounded by
/// the constant 8, which the compiler unrolls.
#[inline(always)]
fn packets_of(count: usize) -> impl Iterator<Item = usize> {
    (0..Q8_BLOCK / PACKET).take_while(move |local| *local < count)
}

/// Packets of super block `b` of a row of `k` values.
#[inline(always)]
fn block_packets(b: usize, k: usize) -> std::ops::Range<usize> {
    8 * b..(8 * b + 8).min(k.div_ceil(PACKET))
}

/// Dense rows of element `E`.
#[derive(Clone, Copy, Debug)]
pub struct DenseRows<E: Dense>(E);

impl<E: Dense> Format for DenseRows<E> {
    #[inline(always)]
    unsafe fn decode<V: Vectors>(row: &Row, p: usize, k: usize, out: &mut [f32; PACKET]) {
        let first = p * PACKET;
        // A whole packet widens in the tier's vectors; a row's tail and a
        // tier without them widen value by value.
        if k - first >= PACKET {
            let values = unsafe { row.start.cast::<E::Storage>().add(first) };
            #[cfg(target_arch = "x86_64")]
            if V::QUANTIZED_DOTS >= 1 {
                for chunk in 0..PACKET / 8 {
                    unsafe {
                        let widened = E::widen_x86(values.add(8 * chunk));
                        std::arch::x86_64::_mm256_storeu_ps(
                            out.as_mut_ptr().add(8 * chunk),
                            widened,
                        );
                    }
                }
                return;
            }
            #[cfg(target_arch = "aarch64")]
            {
                for chunk in 0..PACKET / 8 {
                    unsafe {
                        let [low, high] = E::widen_neon(values.add(8 * chunk));
                        let out = out.as_mut_ptr().add(8 * chunk);
                        std::arch::aarch64::vst1q_f32(out, low);
                        std::arch::aarch64::vst1q_f32(out.add(4), high);
                    }
                }
                return;
            }
        }
        let valid = (k - first).min(PACKET);
        let values =
            unsafe { std::slice::from_raw_parts(row.start.cast::<E::Storage>().add(first), valid) };
        for (target, value) in out.iter_mut().zip(values) {
            *target = E::widen(*value);
        }
        for target in &mut out[valid..] {
            *target = 0.0;
        }
    }

    /// The packet's values widen in the tier's vectors straight into the
    /// products, eight values a multiply-add as [`accumulate`] adds them,
    /// never stored; only a tier without them decodes into `packet`.
    #[inline(always)]
    #[cfg_attr(target_arch = "aarch64", allow(unused_variables))]
    unsafe fn dot_packet<V: Vectors>(
        row: &Row,
        p: usize,
        k: usize,
        lanes: &mut [f32; LANES],
        x: &[f32; PACKET],
        packet: &mut [f32; PACKET],
    ) {
        const { assert!(LANES == 8 && PACKET % LANES == 0, "a lane vector of eight") };
        let values = unsafe { row.start.cast::<E::Storage>().add(p * PACKET) };
        #[cfg(target_arch = "x86_64")]
        if V::QUANTIZED_DOTS >= 1 {
            use std::arch::x86_64::*;
            unsafe {
                let mut sum = _mm256_loadu_ps(lanes.as_ptr());
                for chunk in 0..PACKET / LANES {
                    let at = LANES * chunk;
                    sum = _mm256_fmadd_ps(
                        E::widen_x86(values.add(at)),
                        _mm256_loadu_ps(x.as_ptr().add(at)),
                        sum,
                    );
                }
                _mm256_storeu_ps(lanes.as_mut_ptr(), sum);
            }
            return;
        }
        #[cfg(target_arch = "aarch64")]
        unsafe {
            use std::arch::aarch64::*;
            let (mut low, mut high) = (vld1q_f32(lanes.as_ptr()), vld1q_f32(lanes.as_ptr().add(4)));
            for chunk in 0..PACKET / LANES {
                let at = LANES * chunk;
                let [first, second] = E::widen_neon(values.add(at));
                low = vfmaq_f32(low, first, vld1q_f32(x.as_ptr().add(at)));
                high = vfmaq_f32(high, second, vld1q_f32(x.as_ptr().add(at + 4)));
            }
            vst1q_f32(lanes.as_mut_ptr(), low);
            vst1q_f32(lanes.as_mut_ptr().add(4), high);
            return;
        }
        #[allow(unreachable_code)]
        {
            unsafe { Self::decode::<V>(row, p, k, packet) };
            unsafe { accumulate::<V>(lanes, packet, x) };
        }
    }

    #[inline(always)]
    unsafe fn dot_q8<const R: usize, const M: usize, V: Vectors>(
        rows: &[Row; R],
        b: usize,
        k: usize,
        x: [&Q8Block; M],
    ) -> [[f32; M]; R] {
        let first = b * Q8_BLOCK;
        let valid = (k - first).min(Q8_BLOCK);
        let mut out = [[0.0f32; M]; R];
        if valid == Q8_BLOCK {
            // A whole block: each packet widens once for every activation
            // block and accumulates in the tier's vectors, value `i` into lane
            // `i % LANES` in index order as below.
            for (row, out) in rows.iter().zip(&mut out) {
                let mut lanes = [[0.0f32; LANES]; M];
                for local in 0..Q8_BLOCK / PACKET {
                    let mut values = V::ZERO;
                    let values = V::packet(&mut values);
                    unsafe { Self::decode::<V>(row, b * (Q8_BLOCK / PACKET) + local, k, values) };
                    for (lanes, x) in lanes.iter_mut().zip(x.iter()) {
                        let mut codes = V::ZERO;
                        let codes = V::packet(&mut codes);
                        for (code, x) in codes.iter_mut().zip(&x.codes[PACKET * local..]) {
                            *code = f32::from(*x);
                        }
                        unsafe { accumulate::<V>(lanes, values, codes) };
                    }
                }
                for ((out, lanes), x) in out.iter_mut().zip(lanes).zip(x.iter()) {
                    *out = x.d * combine(lanes);
                }
            }
            return out;
        }
        for (row, out) in rows.iter().zip(&mut out) {
            let values = unsafe {
                std::slice::from_raw_parts(row.start.cast::<E::Storage>().add(first), valid)
            };
            for (out, x) in out.iter_mut().zip(x.iter()) {
                let mut lanes = [0.0f32; LANES];
                for (i, (value, code)) in values.iter().zip(&x.codes).enumerate() {
                    lanes[i % LANES] = E::widen(*value).mul_add(f32::from(*code), lanes[i % LANES]);
                }
                *out = x.d * combine(lanes);
            }
        }
        out
    }
}

/// Where a packet lies: the first byte of its storage group in each plane
/// (`RowPlaneRole` order), resolved once per row and storage group, the
/// group and the packet within it, and its row (whose code planes say where
/// its codes lie, [`At::code`]).
#[derive(Clone, Copy, Debug)]
pub struct At<'a> {
    planes: [*const u8; 4],
    group: usize,
    pub packet: usize,
    row: &'a Row,
}

/// Where one packet's codes of one row lie in a code plane.
#[derive(Clone, Copy, Debug)]
enum CodeBytes {
    /// Together from this byte, in the packet's order (`rows16`'s): every
    /// layout's but a rows8 or an `mma16` codes plane's.
    Packet(*const u8),
    /// A rows8 panel's units from this byte: the row's 4 bytes of each, 32
    /// bytes apart, slot `s` of byte `j` of a unit of `32 / bits` columns
    /// its code of column `4 s + j` (`registry::Layout::Rows8`).
    Panel(*const u8),
    /// An `mma16` codes plane's lanes ([`Lanes`]).
    Lanes(Lanes),
}

/// One packet's codes of one row of an `mma16` codes plane: in each of the
/// four lanes `t` of its row pair (`g`, `g + 8`), the k-block's four steps
/// of 16 columns, a quarter of the lane each, the packet's two (its half
/// `half` of the k-block) steps `2 half + s`, each step a word holding slot
/// `j` of its eight slots at bits `bits * j`, slot `2 b + upper` this row's
/// code of column `16 s + [2t, 2t + 8, 2t + 1, 2t + 9][b]` of step `s`.
#[derive(Clone, Copy, Debug)]
struct Lanes {
    /// Lanes 0 and 1, and lanes 2 and 3, each pair's bytes together.
    pairs: [*const u8; 2],
    /// Bytes of one lane of a k-block (from one lane to the next).
    unit: usize,
    half: usize,
    /// Whether the row is its pair's second (`g + 8`).
    upper: bool,
}

impl Lanes {
    /// The packet's steps of lane `t`.
    #[inline(always)]
    fn lane(self, t: usize) -> *const u8 {
        self.pairs[t / 2].wrapping_add(t % 2 * self.unit + self.half * self.unit / 2)
    }
}

impl<'a> At<'a> {
    /// Packet 0 of storage group `index` of `row`.
    #[inline(always)]
    fn group(row: &'a Row, index: usize) -> Self {
        Self {
            planes: row.group(index),
            group: index,
            packet: 0,
            row,
        }
    }

    /// Where this packet's codes of code plane `role` lie.
    #[inline(always)]
    fn code(self, role: RowPlaneRole) -> CodeBytes {
        let role = role as usize;
        let Some(mma16) = &self.row.mma16 else {
            let bytes = self.planes[role].wrapping_add(self.packet * self.row.packet_steps[role]);
            return if self.row.panel { CodeBytes::Panel(bytes) } else { CodeBytes::Packet(bytes) };
        };
        let plane = &mma16.planes[role];
        if plane.grid {
            // A grid row's k-block: its two packets one after the other.
            let (block, half) = self.block(mma16);
            let first = plane.unit(16 * block + mma16.row);
            return CodeBytes::Packet(first.wrapping_add(half * plane.unit / 2));
        }
        CodeBytes::Lanes(self.lanes_and_high()[role])
    }

    /// The first byte of grid entry `e` (of four) of this packet, entries of
    /// `bytes` bytes: one after another in a grids plane, a `records`
    /// tile's at `record_entry`.
    #[inline(always)]
    fn entry(self, e: usize, bytes: usize) -> *const u8 {
        let Some(row) = self.row.record else {
            return packet_bytes(self, RowPlaneRole::Codes).wrapping_add(bytes * e);
        };
        // The packet's half of its k-block holds the k-block's entries
        // 4 half..4 half + 4.
        let (block, half) = (self.packet / 2, self.packet % 2);
        let within = registry::PackedRowLayout::record_entry(row as u64, block as u64, (4 * half + e) as u64);
        self.planes[RowPlaneRole::Codes as usize].wrapping_add(within as usize)
    }

    /// The `mma16` k-block holding this packet, and the packet's half of it.
    #[inline(always)]
    fn block(self, mma16: &Mma16Row) -> (usize, usize) {
        let packet = mma16.first_packet + self.group * mma16.group_packets + self.packet;
        (packet / 2, packet % 2)
    }

    /// Whether the codes lie in `mma16` lanes (`CodeBytes::Lanes`): the
    /// row's layout is `mma16` and its code plane holds codes, not grids.
    #[inline(always)]
    fn in_lanes(self) -> bool {
        self.row.mma16.as_ref().is_some_and(|mma16| !mma16.planes[0].grid)
    }

    /// Where this packet's codes lie in an `mma16` codes plane and its high
    /// plane, which place their lanes alike (as many a row): one placement
    /// for both.
    #[inline(always)]
    fn lanes_and_high(self) -> [Lanes; 2] {
        let mma16 = self.row.mma16.as_ref().expect("an mma16 row's lanes");
        let (block, half) = self.block(mma16);
        // A lane holds four steps, the packet's two its half; a k-block's
        // lanes 4g..4g + 4 are the pair's (two lanes of a pair never lie
        // apart, a row's payload being whole lane pairs).
        let first = 32 * block + 4 * (mma16.row % 8);
        let [low, high] = &mma16.planes;
        let places = [low.place(first), low.place(first + 2)];
        let lanes = |plane: &Mma16Plane| Lanes {
            pairs: places.map(|place| plane.at(place)),
            unit: plane.unit,
            half,
            upper: mma16.row >= 8,
        };
        [lanes(low), lanes(high)]
    }

    /// Packet `p` of `row`, of `STORAGE`-value storage groups.
    #[inline(always)]
    fn of<const STORAGE: usize>(row: &'a Row, p: usize) -> Self {
        let packets = STORAGE / PACKET;
        Self::group(row, p / packets).packet(p % packets)
    }

    /// Packet `packet` of the same storage group.
    #[inline(always)]
    fn packet(self, packet: usize) -> Self {
        Self { packet, ..self }
    }

    /// The storage group's first byte in plane `role`.
    #[inline(always)]
    fn plane(self, role: RowPlaneRole) -> *const u8 {
        self.planes[role as usize]
    }

    /// The byte holding the first bit of this row's record `record` of the
    /// group plane `role`, records of `bits` bits one after another
    /// counting across its listed planes (`registry::PackedRowLayout::
    /// record_bit`), and that bit's place in it: a rows8 span holds each
    /// record of its eight rows in turn.
    #[inline(always)]
    fn field(self, role: RowPlaneRole, record: usize, bits: usize) -> (*const u8, usize) {
        let bit = match self.row.tile {
            Some((_, r)) => (8 * record + r) * bits,
            None => record * bits,
        };
        (self.plane(role).wrapping_add(bit / 8), bit % 8)
    }

    /// Coefficient group of `GROUP` values (within the storage group) of
    /// this packet's half `half`.
    #[inline(always)]
    fn local<const GROUP: usize>(self, half: usize) -> usize {
        (PACKET * self.packet + 16 * half) / GROUP
    }
}

/// The packets of activation block `b` of a row of `STORAGE`-value storage
/// groups (a divisor of the block): its first storage group and the bytes
/// from one group to the next in each plane.
#[derive(Clone, Copy, Debug)]
struct Block<'a> {
    first: At<'a>,
    steps: [usize; 4],
}

impl<'a> Block<'a> {
    #[inline(always)]
    fn of<const STORAGE: usize>(row: &'a Row, b: usize) -> Self {
        const {
            assert!(
                Q8_BLOCK % STORAGE == 0 && STORAGE % PACKET == 0,
                "storage groups divide a block"
            )
        };
        Self {
            first: At::group(row, b * (Q8_BLOCK / STORAGE)),
            steps: row.steps,
        }
    }

    /// Packet `local` of the block.
    #[inline(always)]
    fn at<const STORAGE: usize>(&self, local: usize) -> At<'a> {
        let packets = STORAGE / PACKET;
        let group = local / packets;
        let mut at = self.first.packet(local % packets);
        for plane in 0..4 {
            at.planes[plane] = at.planes[plane].wrapping_add(group * self.steps[plane]);
        }
        at.group += group;
        at
    }
}

/// The interpretations of codes (`registry::CodeInterpretation`): a raw
/// `bits`-bit code's exact integer value. Which interpretations a code plane
/// reads are the [`Codes`] implementations.
pub mod codes {
    /// The code itself.
    #[derive(Clone, Copy, Debug)]
    pub struct Unsigned;
    /// Two's complement.
    #[derive(Clone, Copy, Debug)]
    pub struct Twos;
    /// The code minus `C`.
    #[derive(Clone, Copy, Debug)]
    pub struct Offset<const C: i32>;
    /// The odd value `2 (c - K) + 1`.
    #[derive(Clone, Copy, Debug)]
    pub struct Odd<const K: i32>;
    /// Value `c` of codebook `ID` (its position in `tables::CODEBOOKS`).
    #[derive(Clone, Copy, Debug)]
    pub struct Table<const ID: usize>;
}
use codes::{Odd, Offset, Table, Twos, Unsigned};

/// A raw code's exact integer value.
pub trait Interpretation: Copy + Send + Sync + 'static {
    fn value<const BITS: u32>(raw: u32) -> i32;

    /// [`Interpretation::value`] of each i32 lane of `raw`.
    ///
    /// # Safety
    /// The caller's tier has 256-bit integer vectors.
    #[cfg(target_arch = "x86_64")]
    #[inline(always)]
    unsafe fn values8<const BITS: u32>(raw: std::arch::x86_64::__m256i) -> std::arch::x86_64::__m256i {
        use std::arch::x86_64::*;
        let mut lanes = [0u32; 8];
        unsafe { _mm256_storeu_si256(lanes.as_mut_ptr().cast(), raw) };
        let values = lanes.map(|lane| Self::value::<BITS>(lane));
        unsafe { _mm256_loadu_si256(values.as_ptr().cast()) }
    }
}

impl Interpretation for Unsigned {
    #[inline(always)]
    fn value<const BITS: u32>(raw: u32) -> i32 {
        raw as i32
    }

    #[cfg(target_arch = "x86_64")]
    #[inline(always)]
    unsafe fn values8<const BITS: u32>(raw: std::arch::x86_64::__m256i) -> std::arch::x86_64::__m256i {
        raw
    }
}

impl Interpretation for Twos {
    #[inline(always)]
    fn value<const BITS: u32>(raw: u32) -> i32 {
        ((raw << (32 - BITS)) as i32) >> (32 - BITS)
    }

    #[cfg(target_arch = "x86_64")]
    #[inline(always)]
    unsafe fn values8<const BITS: u32>(raw: std::arch::x86_64::__m256i) -> std::arch::x86_64::__m256i {
        use std::arch::x86_64::*;
        let shift = _mm_cvtsi32_si128(32 - BITS as i32);
        unsafe { _mm256_sra_epi32(_mm256_sll_epi32(raw, shift), shift) }
    }
}

impl<const C: i32> Interpretation for Offset<C> {
    #[inline(always)]
    fn value<const BITS: u32>(raw: u32) -> i32 {
        raw as i32 - C
    }

    #[cfg(target_arch = "x86_64")]
    #[inline(always)]
    unsafe fn values8<const BITS: u32>(raw: std::arch::x86_64::__m256i) -> std::arch::x86_64::__m256i {
        use std::arch::x86_64::*;
        unsafe { _mm256_sub_epi32(raw, _mm256_set1_epi32(C)) }
    }
}

impl<const K: i32> Interpretation for Odd<K> {
    #[inline(always)]
    fn value<const BITS: u32>(raw: u32) -> i32 {
        2 * (raw as i32 - K) + 1
    }
}

impl<const ID: usize> Table<ID> {
    const VALUES: [i8; 16] = {
        let values = tables::CODEBOOKS[ID].values;
        assert!(values.len() == 16, "a 4-bit codebook");
        let mut out = [0i8; 16];
        let mut i = 0;
        while i < 16 {
            out[i] = values[i];
            i += 1;
        }
        out
    };
}

impl<const ID: usize> Interpretation for Table<ID> {
    #[inline(always)]
    fn value<const BITS: u32>(raw: u32) -> i32 {
        i32::from(Self::VALUES[raw as usize])
    }
}

/// How a code component's codes lie in rows8 panels
/// (`registry::Layout::Rows8`): the bits of its low and high planes, and
/// a raw code's value, `values[raw]` with a table (at most four bits),
/// else `raw + offset` (as bytes).
#[derive(Clone, Copy, Debug)]
pub struct PanelCodes {
    pub low: u32,
    pub high: u32,
    pub table: Option<[i8; 16]>,
    pub offset: i8,
}

impl PanelCodes {
    /// Codes of `low` and `high` bits whose value is `raw + offset`.
    const fn offset(low: u32, high: u32, offset: i32) -> Option<Self> {
        Some(Self { low, high, table: None, offset: offset as i8 })
    }

    /// The value of raw code `raw`.
    #[inline(always)]
    fn value(self, raw: u8) -> i8 {
        match self.table {
            Some(table) => table[usize::from(raw)],
            None => (raw as i8).wrapping_add(self.offset),
        }
    }
}

/// A packet's 32 exact integer values: a code component.
pub trait Codes: Copy + Send + Sync + 'static {
    /// The least and the greatest value: the registry's code domain
    /// (`DecodeAxes::codes`), a component's const parameters.
    const LEAST: i32;
    const GREATEST: i32;

    /// How the codes lie in rows8 panels; None for grid codes, whose
    /// entries are none.
    const PANEL: Option<PanelCodes> = None;

    /// Where an `mma16` tile's codes are 4-bit lanes (with a high plane of
    /// `HIGH` bits a code, at most 2) whose joined raw code `r` is the value
    /// `r + offset`: `(HIGH, offset)`, which a 16-row tile's quantized dot
    /// reads with no unpermuting (`Hier::dot_tile16`). None for any other
    /// codes.
    const NIBBLE_LANES: Option<(u32, i32)> = None;

    /// The values of packet `at`, in element order.
    ///
    /// # Safety
    /// `at` is a packet of a row.
    unsafe fn values(at: At) -> [i8; PACKET];

    /// The values in registers, values `0..16` then `16..32`.
    ///
    /// # Safety
    /// As [`Codes::values`].
    #[cfg(target_arch = "aarch64")]
    #[inline(always)]
    unsafe fn neon(at: At) -> [std::arch::aarch64::int8x16_t; 2] {
        use std::arch::aarch64::*;
        let values = unsafe { Self::values(at) };
        unsafe { [vld1q_s8(values.as_ptr()), vld1q_s8(values.as_ptr().add(16))] }
    }

    /// [`Codes::neon`] of packet `at` and the next, `at` the first of a
    /// k-block's two (an even packet of its storage group, which holds whole
    /// k-blocks): one read where a layout keeps a k-block's together
    /// (`mma16`).
    ///
    /// # Safety
    /// As [`Codes::values`], of both packets.
    #[cfg(target_arch = "aarch64")]
    #[inline(always)]
    unsafe fn neon_pair(at: At) -> [[std::arch::aarch64::int8x16_t; 2]; 2] {
        unsafe { [Self::neon(at), Self::neon(at.packet(at.packet + 1))] }
    }

    /// [`Codes::neon_pair`] of the same packet of rows `g` and `g + 8` of
    /// an `mma16` tile (`Row::partners`): one read of their shared bytes
    /// where the codes lie in its lanes.
    ///
    /// # Safety
    /// As [`Codes::neon_pair`], of both rows.
    #[cfg(target_arch = "aarch64")]
    #[inline(always)]
    unsafe fn neon_partners(g: At, partner: At) -> [[[std::arch::aarch64::int8x16_t; 2]; 2]; 2] {
        unsafe { [Self::neon_pair(g), Self::neon_pair(partner)] }
    }

    /// The values in one register.
    ///
    /// # Safety
    /// As [`Codes::values`]; the caller's tier has 256-bit integer vectors.
    #[cfg(target_arch = "x86_64")]
    #[inline(always)]
    unsafe fn x86(at: At) -> std::arch::x86_64::__m256i {
        let values = unsafe { Self::values(at) };
        unsafe { packed_x86::load(values.as_ptr()) }
    }

    /// [`Codes::x86`] of packet `at` and the next, as [`Codes::neon_pair`].
    ///
    /// # Safety
    /// As [`Codes::x86`], of both packets.
    #[cfg(target_arch = "x86_64")]
    #[inline(always)]
    unsafe fn x86_pair(at: At) -> [std::arch::x86_64::__m256i; 2] {
        unsafe { [Self::x86(at), Self::x86(at.packet(at.packet + 1))] }
    }

    /// [`Codes::x86_pair`] of rows `g` and `g + 8`, as
    /// [`Codes::neon_partners`].
    ///
    /// # Safety
    /// As [`Codes::x86_pair`], of both rows.
    #[cfg(target_arch = "x86_64")]
    #[inline(always)]
    unsafe fn x86_partners(g: At, partner: At) -> [[std::arch::x86_64::__m256i; 2]; 2] {
        unsafe { [Self::x86_pair(g), Self::x86_pair(partner)] }
    }

    /// The values minus [`Codes::LEAST`] in one register, as unsigned bytes.
    ///
    /// # Safety
    /// As [`Codes::x86`].
    #[cfg(target_arch = "x86_64")]
    #[inline(always)]
    unsafe fn x86_shifted(at: At) -> std::arch::x86_64::__m256i {
        use std::arch::x86_64::*;
        unsafe { _mm256_sub_epi8(Self::x86(at), _mm256_set1_epi8(Self::LEAST as i8)) }
    }
}

/// Codes in a `LOW`-bit low plane and a `HIGH`-bit high plane
/// (`registry::code_planes`), read through interpretation `I`, whose values
/// are the registry's code domain `LEAST..=GREATEST` (`DecodeAxes::codes`):
/// code `i` of a packet is bits `LOW i ..` of the low plane (the low nibble
/// of byte `i / 2` first for 4-bit codes; byte `i` of an 8-bit plane) joined
/// with bits `HIGH i ..` of the high plane shifted to bit `LOW`.
#[derive(Clone, Copy, Debug)]
pub struct Planes<const LOW: u32, const HIGH: u32, I, const LEAST: i32, const GREATEST: i32>(
    PhantomData<I>,
);

/// Lane `t` and slot pair `b` holding column `k` of a 16-column step of an
/// `mma16` codes plane (`Lanes`).
const fn lane_slot(k: usize) -> (usize, usize) {
    ((k % 8) / 2, k / 8 + 2 * (k % 2))
}

/// The packet's raw `bits`-bit codes (`bits` dividing 8) of an `mma16`
/// codes plane, in element order.
#[inline(always)]
unsafe fn lane_codes(lanes: Lanes, bits: usize) -> [u8; PACKET] {
    let mask = ((1u16 << bits) - 1) as u8;
    let mut codes = [0u8; PACKET];
    for (c, code) in codes.iter_mut().enumerate() {
        let (step, (t, b)) = (c / 16, lane_slot(c % 16));
        let bit = bits * (2 * b + lanes.upper as usize);
        let byte = unsafe { *lanes.lane(t).add(bits * step + bit / 8) };
        *code = (byte >> (bit % 8)) & mask;
    }
    codes
}

/// The bytes a code plane `at` lies in keeps together (`At::code`).
#[inline(always)]
fn packet_bytes(at: At, role: RowPlaneRole) -> *const u8 {
    match at.code(role) {
        CodeBytes::Packet(bytes) => bytes,
        CodeBytes::Lanes(_) | CodeBytes::Panel(_) => {
            unreachable!("a code plane keeps its packet together or none does")
        }
    }
}

/// The packet's raw `bits`-bit codes of a rows8 panel from `units`
/// (`CodeBytes::Panel`), in element order.
#[inline(always)]
unsafe fn panel_codes(units: *const u8, bits: usize) -> [u8; PACKET] {
    let (chunk, mask) = (32 / bits, ((1u16 << bits) - 1) as u8);
    // A loop, not `array::from_fn`: that compiles out of line.
    let mut codes = [0u8; PACKET];
    for (c, code) in codes.iter_mut().enumerate() {
        let byte = unsafe { *units.add(c / chunk * 32 + c % 4) };
        *code = (byte >> (c % chunk / 4 * bits)) & mask;
    }
    codes
}

impl<const LOW: u32, const HIGH: u32, I, const LEAST: i32, const GREATEST: i32>
    Planes<LOW, HIGH, I, LEAST, GREATEST>
{
    /// The code domain, whose values fit a signed byte.
    const DOMAIN: (i32, i32) = {
        assert!(
            i8::MIN as i32 <= LEAST && LEAST <= GREATEST && GREATEST <= i8::MAX as i32,
            "values fit a signed byte"
        );
        (LEAST, GREATEST)
    };

    /// The packet's raw codes.
    #[inline(always)]
    unsafe fn raw(at: At) -> [u8; PACKET] {
        const {
            assert!(
                matches!(LOW, 1 | 2 | 4) && HIGH <= 2,
                "a 1-, 2- or 4-bit low plane and a high plane of at most 2 bits"
            )
        };
        let mut codes = [0u8; PACKET];
        if at.in_lanes() {
            let [low, high] = at.lanes_and_high();
            codes = unsafe { lane_codes(low, LOW as usize) };
            if HIGH > 0 {
                let high = unsafe { lane_codes(high, HIGH as usize) };
                for (code, high) in codes.iter_mut().zip(high) {
                    *code |= high << LOW;
                }
            }
            return codes;
        }
        if let CodeBytes::Panel(low) = at.code(RowPlaneRole::Codes) {
            codes = unsafe { panel_codes(low, LOW as usize) };
            if let (true, CodeBytes::Panel(high)) = (HIGH > 0, at.code(RowPlaneRole::High)) {
                for (code, high) in codes.iter_mut().zip(unsafe { panel_codes(high, HIGH as usize) }) {
                    *code |= high << LOW;
                }
            }
            return codes;
        }
        let mut bytes = [0u8; 16];
        unsafe {
            std::ptr::copy_nonoverlapping(
                packet_bytes(at, RowPlaneRole::Codes),
                bytes.as_mut_ptr(),
                4 * LOW as usize,
            )
        };
        // Code `(8 / LOW) i + j` is bits `LOW j ..` of byte `i`.
        if LOW == 4 {
            for (i, byte) in bytes.iter().enumerate() {
                codes[2 * i] = byte & 15;
                codes[2 * i + 1] = byte >> 4;
            }
        } else if LOW == 2 {
            for (i, byte) in bytes[..8].iter().enumerate() {
                for j in 0..4 {
                    codes[4 * i + j] = (byte >> (2 * j)) & 3;
                }
            }
        } else {
            for (i, byte) in bytes[..4].iter().enumerate() {
                for j in 0..8 {
                    codes[8 * i + j] = (byte >> j) & 1;
                }
            }
        }
        if HIGH > 0 {
            let high_bytes = packet_bytes(at, RowPlaneRole::High);
            let high: u64 = match HIGH {
                1 => u64::from(unsafe { read::<u32>(high_bytes) }),
                _ => unsafe { read(high_bytes) },
            };
            let mask = (1u64 << HIGH) - 1;
            for (i, code) in codes.iter_mut().enumerate() {
                *code |= (((high >> (HIGH as usize * i)) & mask) as u8) << LOW;
            }
        }
        codes
    }

    #[cfg(target_arch = "aarch64")]
    #[inline(always)]
    unsafe fn raw_neon(at: At) -> [std::arch::aarch64::int8x16_t; 2] {
        const {
            assert!(
                HIGH <= 1 || LOW == 4,
                "a 2-bit high plane joins a nibble low plane"
            )
        };
        if at.in_lanes() {
            let [low, high] = at.lanes_and_high();
            let codes = unsafe { packed_neon::lane_codes::<LOW>(low) };
            return match HIGH {
                0 => codes,
                _ => unsafe { packed_neon::with_high::<LOW>(codes, high, HIGH) },
            };
        }
        if let CodeBytes::Panel(low) = at.code(RowPlaneRole::Codes) {
            let codes = unsafe { packed_neon::panel::<LOW>(low) };
            let CodeBytes::Panel(high) = at.code(RowPlaneRole::High) else { return codes };
            return match HIGH {
                0 => codes,
                _ => {
                    let high = unsafe { packed_neon::panel::<HIGH>(high) };
                    [0, 1].map(|half| unsafe {
                        use std::arch::aarch64::*;
                        vorrq_s8(codes[half], vshlq_s8(high[half], vdupq_n_s8(LOW as i8)))
                    })
                }
            };
        }
        let low = packet_bytes(at, RowPlaneRole::Codes);
        let high = || packet_bytes(at, RowPlaneRole::High);
        match HIGH {
            0 => unsafe { packed_neon::codes::<LOW>(low, 0) },
            1 => unsafe { packed_neon::codes::<LOW>(low, read(high())) },
            _ => unsafe { packed_neon::nibbles2(low, high()) },
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[inline(always)]
    unsafe fn raw_x86(at: At) -> std::arch::x86_64::__m256i {
        const {
            assert!(
                HIGH <= 1 || LOW == 4,
                "a 2-bit high plane joins a nibble low plane"
            )
        };
        if at.in_lanes() {
            let [low, high] = at.lanes_and_high();
            let codes = unsafe { packed_x86::lane_codes::<LOW>(low) };
            return match HIGH {
                0 => codes,
                _ => unsafe { packed_x86::with_high::<LOW>(codes, high, HIGH) },
            };
        }
        if let CodeBytes::Panel(low) = at.code(RowPlaneRole::Codes) {
            let codes = unsafe { packed_x86::panel::<LOW>(low) };
            return match (HIGH, at.code(RowPlaneRole::High)) {
                (1 | 2, CodeBytes::Panel(high)) => unsafe {
                    use std::arch::x86_64::*;
                    let high = packed_x86::panel::<HIGH>(high);
                    _mm256_or_si256(codes, _mm256_sll_epi16(high, _mm_cvtsi32_si128(LOW as i32)))
                },
                _ => codes,
            };
        }
        let low = packet_bytes(at, RowPlaneRole::Codes);
        let high = || packet_bytes(at, RowPlaneRole::High);
        match HIGH {
            0 => unsafe { packed_x86::codes::<LOW>(low) },
            1 => unsafe { packed_x86::codes1::<LOW>(low, read(high())) },
            _ => unsafe { packed_x86::nibbles2(low, high()) },
        }
    }

    /// [`Self::raw_neon`] of packet `at` and the next (`Codes::neon_pair`):
    /// an `mma16` k-block's nibble lanes read once (lanes of narrower codes
    /// are read packet by packet, [`packed_neon::lane_codes`]).
    #[cfg(target_arch = "aarch64")]
    #[inline(always)]
    unsafe fn raw_neon_pair(at: At) -> [[std::arch::aarch64::int8x16_t; 2]; 2] {
        if !at.in_lanes() || LOW != 4 {
            return unsafe { [Self::raw_neon(at), Self::raw_neon(at.packet(at.packet + 1))] };
        }
        let [low, high] = at.lanes_and_high();
        let codes = unsafe { packed_neon::block_nibbles(low) };
        match HIGH {
            0 => codes,
            _ => unsafe {
                [
                    packed_neon::with_high::<LOW>(codes[0], Lanes { half: 0, ..high }, HIGH),
                    packed_neon::with_high::<LOW>(codes[1], Lanes { half: 1, ..high }, HIGH),
                ]
            },
        }
    }

    /// [`Self::raw_x86`] of packet `at` and the next, as
    /// [`Self::raw_neon_pair`].
    #[cfg(target_arch = "x86_64")]
    #[inline(always)]
    unsafe fn raw_x86_pair(at: At) -> [std::arch::x86_64::__m256i; 2] {
        if !at.in_lanes() || LOW != 4 {
            return unsafe { [Self::raw_x86(at), Self::raw_x86(at.packet(at.packet + 1))] };
        }
        let [low, high] = at.lanes_and_high();
        match HIGH {
            0 => unsafe { packed_x86::block_nibbles(low) },
            _ => unsafe {
                let codes = packed_x86::block_nibbles(low);
                [
                    packed_x86::with_high::<LOW>(codes[0], Lanes { half: 0, ..high }, HIGH),
                    packed_x86::with_high::<LOW>(codes[1], Lanes { half: 1, ..high }, HIGH),
                ]
            },
        }
    }

    /// [`Self::raw_neon_pair`] of rows `g` and `g + 8` (`Codes::neon_partners`):
    /// their shared nibble lanes read once.
    #[cfg(target_arch = "aarch64")]
    #[inline(always)]
    unsafe fn raw_neon_partners(g: At, partner: At) -> [[[std::arch::aarch64::int8x16_t; 2]; 2]; 2] {
        if !g.in_lanes() || LOW != 4 {
            return unsafe { [Self::raw_neon_pair(g), Self::raw_neon_pair(partner)] };
        }
        let [low, high] = g.lanes_and_high();
        let codes = unsafe { packed_neon::block_partners(low) };
        match HIGH {
            0 => codes,
            _ => {
                let [_, partner_high] = partner.lanes_and_high();
                unsafe {
                    [
                        [
                            packed_neon::with_high::<LOW>(codes[0][0], Lanes { half: 0, ..high }, HIGH),
                            packed_neon::with_high::<LOW>(codes[0][1], Lanes { half: 1, ..high }, HIGH),
                        ],
                        [
                            packed_neon::with_high::<LOW>(codes[1][0], Lanes { half: 0, ..partner_high }, HIGH),
                            packed_neon::with_high::<LOW>(codes[1][1], Lanes { half: 1, ..partner_high }, HIGH),
                        ],
                    ]
                }
            }
        }
    }

    /// [`Self::raw_x86_pair`] of rows `g` and `g + 8`, as
    /// [`Self::raw_neon_partners`].
    #[cfg(target_arch = "x86_64")]
    #[inline(always)]
    unsafe fn raw_x86_partners(g: At, partner: At) -> [[std::arch::x86_64::__m256i; 2]; 2] {
        if !g.in_lanes() || LOW != 4 {
            return unsafe { [Self::raw_x86_pair(g), Self::raw_x86_pair(partner)] };
        }
        let [low, high] = g.lanes_and_high();
        let codes = unsafe { packed_x86::block_partners(low) };
        match HIGH {
            0 => codes,
            _ => {
                let [_, partner_high] = partner.lanes_and_high();
                unsafe {
                    [
                        [
                            packed_x86::with_high::<LOW>(codes[0][0], Lanes { half: 0, ..high }, HIGH),
                            packed_x86::with_high::<LOW>(codes[0][1], Lanes { half: 1, ..high }, HIGH),
                        ],
                        [
                            packed_x86::with_high::<LOW>(codes[1][0], Lanes { half: 0, ..partner_high }, HIGH),
                            packed_x86::with_high::<LOW>(codes[1][1], Lanes { half: 1, ..partner_high }, HIGH),
                        ],
                    ]
                }
            }
        }
    }
}

impl<const LOW: u32, const HIGH: u32, const LEAST: i32, const GREATEST: i32> Codes
    for Planes<LOW, HIGH, Unsigned, LEAST, GREATEST>
{
    const LEAST: i32 = Self::DOMAIN.0;
    const GREATEST: i32 = Self::DOMAIN.1;
    const NIBBLE_LANES: Option<(u32, i32)> = if LOW == 4 && HIGH <= 2 { Some((HIGH, 0)) } else { None };
    const PANEL: Option<PanelCodes> = PanelCodes::offset(LOW, HIGH, 0);

    #[inline(always)]
    unsafe fn values(at: At) -> [i8; PACKET] {
        let raw = unsafe { Self::raw(at) };
        let mut values = [0i8; PACKET];
        for i in 0..PACKET {
            values[i] = raw[i] as i8;
        }
        values
    }

    #[cfg(target_arch = "aarch64")]
    #[inline(always)]
    unsafe fn neon(at: At) -> [std::arch::aarch64::int8x16_t; 2] {
        unsafe { Self::raw_neon(at) }
    }

    #[cfg(target_arch = "aarch64")]
    #[inline(always)]
    unsafe fn neon_pair(at: At) -> [[std::arch::aarch64::int8x16_t; 2]; 2] {
        unsafe { Self::raw_neon_pair(at) }
    }

    #[cfg(target_arch = "aarch64")]
    #[inline(always)]
    unsafe fn neon_partners(g: At, partner: At) -> [[[std::arch::aarch64::int8x16_t; 2]; 2]; 2] {
        unsafe { Self::raw_neon_partners(g, partner) }
    }

    #[cfg(target_arch = "x86_64")]
    #[inline(always)]
    unsafe fn x86(at: At) -> std::arch::x86_64::__m256i {
        unsafe { Self::raw_x86(at) }
    }

    #[cfg(target_arch = "x86_64")]
    #[inline(always)]
    unsafe fn x86_pair(at: At) -> [std::arch::x86_64::__m256i; 2] {
        unsafe { Self::raw_x86_pair(at) }
    }

    #[cfg(target_arch = "x86_64")]
    #[inline(always)]
    unsafe fn x86_partners(g: At, partner: At) -> [[std::arch::x86_64::__m256i; 2]; 2] {
        unsafe { Self::raw_x86_partners(g, partner) }
    }

    #[cfg(target_arch = "x86_64")]
    #[inline(always)]
    unsafe fn x86_shifted(at: At) -> std::arch::x86_64::__m256i {
        unsafe { Self::raw_x86(at) }
    }
}

impl<const LOW: u32, const HIGH: u32, const C: i32, const LEAST: i32, const GREATEST: i32> Codes
    for Planes<LOW, HIGH, Offset<C>, LEAST, GREATEST>
{
    const LEAST: i32 = Self::DOMAIN.0;
    const GREATEST: i32 = Self::DOMAIN.1;
    const NIBBLE_LANES: Option<(u32, i32)> = if LOW == 4 && HIGH <= 2 { Some((HIGH, -C)) } else { None };
    const PANEL: Option<PanelCodes> = PanelCodes::offset(LOW, HIGH, -C);

    #[inline(always)]
    unsafe fn values(at: At) -> [i8; PACKET] {
        let raw = unsafe { Self::raw(at) };
        let mut values = [0i8; PACKET];
        for i in 0..PACKET {
            values[i] = (i32::from(raw[i]) - C) as i8;
        }
        values
    }

    #[cfg(target_arch = "aarch64")]
    #[inline(always)]
    unsafe fn neon(at: At) -> [std::arch::aarch64::int8x16_t; 2] {
        use std::arch::aarch64::*;
        let offset = unsafe { vdupq_n_s8(C as i8) };
        unsafe { Self::raw_neon(at) }.map(|codes| unsafe { vsubq_s8(codes, offset) })
    }

    #[cfg(target_arch = "aarch64")]
    #[inline(always)]
    unsafe fn neon_pair(at: At) -> [[std::arch::aarch64::int8x16_t; 2]; 2] {
        use std::arch::aarch64::*;
        let offset = unsafe { vdupq_n_s8(C as i8) };
        unsafe { Self::raw_neon_pair(at) }.map(|packet| packet.map(|codes| unsafe { vsubq_s8(codes, offset) }))
    }

    #[cfg(target_arch = "aarch64")]
    #[inline(always)]
    unsafe fn neon_partners(g: At, partner: At) -> [[[std::arch::aarch64::int8x16_t; 2]; 2]; 2] {
        use std::arch::aarch64::*;
        let offset = unsafe { vdupq_n_s8(C as i8) };
        let mut codes = unsafe { Self::raw_neon_partners(g, partner) };
        for row in &mut codes {
            for packet in row {
                for half in packet {
                    *half = unsafe { vsubq_s8(*half, offset) };
                }
            }
        }
        codes
    }

    #[cfg(target_arch = "x86_64")]
    #[inline(always)]
    unsafe fn x86(at: At) -> std::arch::x86_64::__m256i {
        use std::arch::x86_64::*;
        let offset = _mm256_set1_epi8(C as i8);
        unsafe { _mm256_sub_epi8(Self::raw_x86(at), offset) }
    }

    #[cfg(target_arch = "x86_64")]
    #[inline(always)]
    unsafe fn x86_pair(at: At) -> [std::arch::x86_64::__m256i; 2] {
        use std::arch::x86_64::*;
        let offset = _mm256_set1_epi8(C as i8);
        let [first, second] = unsafe { Self::raw_x86_pair(at) };
        unsafe { [_mm256_sub_epi8(first, offset), _mm256_sub_epi8(second, offset)] }
    }

    #[cfg(target_arch = "x86_64")]
    #[inline(always)]
    unsafe fn x86_partners(g: At, partner: At) -> [[std::arch::x86_64::__m256i; 2]; 2] {
        use std::arch::x86_64::*;
        let offset = _mm256_set1_epi8(C as i8);
        let [[a, b], [c, d]] = unsafe { Self::raw_x86_partners(g, partner) };
        unsafe {
            [
                [_mm256_sub_epi8(a, offset), _mm256_sub_epi8(b, offset)],
                [_mm256_sub_epi8(c, offset), _mm256_sub_epi8(d, offset)],
            ]
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[inline(always)]
    unsafe fn x86_shifted(at: At) -> std::arch::x86_64::__m256i {
        unsafe { Self::raw_x86(at) }
    }
}

/// Odd codes' values from their raw codes: `2 * code + 1 - 2 * K`. (A
/// function, not a closure: a closure would not take the caller's
/// features, and its intrinsics would compile as calls.)
#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn odd_x86<const K: i32>(codes: std::arch::x86_64::__m256i) -> std::arch::x86_64::__m256i {
    use std::arch::x86_64::*;
    unsafe { _mm256_add_epi8(_mm256_add_epi8(codes, codes), _mm256_set1_epi8((1 - 2 * K) as i8)) }
}

impl<const LOW: u32, const HIGH: u32, const K: i32, const LEAST: i32, const GREATEST: i32> Codes
    for Planes<LOW, HIGH, Odd<K>, LEAST, GREATEST>
{
    const LEAST: i32 = Self::DOMAIN.0;
    const GREATEST: i32 = Self::DOMAIN.1;
    const PANEL: Option<PanelCodes> = {
        assert!(LOW + HIGH <= 4, "odd codes of at most four bits");
        let mut table = [0i8; 16];
        let mut raw = 0;
        while raw < 16 {
            table[raw] = (2 * (raw as i32 - K) + 1) as i8;
            raw += 1;
        }
        Some(PanelCodes { low: LOW, high: HIGH, table: Some(table), offset: 0 })
    };

    #[inline(always)]
    unsafe fn values(at: At) -> [i8; PACKET] {
        let raw = unsafe { Self::raw(at) };
        let mut values = [0i8; PACKET];
        for i in 0..PACKET {
            values[i] = Odd::<K>::value::<0>(u32::from(raw[i])) as i8;
        }
        values
    }

    // `2 (c - K) + 1` is the doubled code plus `1 - 2 K`.
    #[cfg(target_arch = "aarch64")]
    #[inline(always)]
    unsafe fn neon(at: At) -> [std::arch::aarch64::int8x16_t; 2] {
        use std::arch::aarch64::*;
        let offset = unsafe { vdupq_n_s8((1 - 2 * K) as i8) };
        unsafe { Self::raw_neon(at) }
            .map(|codes| unsafe { vaddq_s8(vaddq_s8(codes, codes), offset) })
    }

    #[cfg(target_arch = "aarch64")]
    #[inline(always)]
    unsafe fn neon_pair(at: At) -> [[std::arch::aarch64::int8x16_t; 2]; 2] {
        use std::arch::aarch64::*;
        let offset = unsafe { vdupq_n_s8((1 - 2 * K) as i8) };
        unsafe { Self::raw_neon_pair(at) }
            .map(|packet| packet.map(|codes| unsafe { vaddq_s8(vaddq_s8(codes, codes), offset) }))
    }

    #[cfg(target_arch = "aarch64")]
    #[inline(always)]
    unsafe fn neon_partners(g: At, partner: At) -> [[[std::arch::aarch64::int8x16_t; 2]; 2]; 2] {
        use std::arch::aarch64::*;
        let offset = unsafe { vdupq_n_s8((1 - 2 * K) as i8) };
        unsafe { Self::raw_neon_partners(g, partner) }.map(|row| {
            row.map(|packet| packet.map(|codes| unsafe { vaddq_s8(vaddq_s8(codes, codes), offset) }))
        })
    }

    #[cfg(target_arch = "x86_64")]
    #[inline(always)]
    unsafe fn x86(at: At) -> std::arch::x86_64::__m256i {
        unsafe { odd_x86::<K>(Self::raw_x86(at)) }
    }

    #[cfg(target_arch = "x86_64")]
    #[inline(always)]
    unsafe fn x86_pair(at: At) -> [std::arch::x86_64::__m256i; 2] {
        let [first, second] = unsafe { Self::raw_x86_pair(at) };
        unsafe { [odd_x86::<K>(first), odd_x86::<K>(second)] }
    }

    #[cfg(target_arch = "x86_64")]
    #[inline(always)]
    unsafe fn x86_partners(g: At, partner: At) -> [[std::arch::x86_64::__m256i; 2]; 2] {
        let [[a, b], [c, d]] = unsafe { Self::raw_x86_partners(g, partner) };
        unsafe { [[odd_x86::<K>(a), odd_x86::<K>(b)], [odd_x86::<K>(c), odd_x86::<K>(d)]] }
    }

    #[cfg(target_arch = "x86_64")]
    #[inline(always)]
    unsafe fn x86_shifted(at: At) -> std::arch::x86_64::__m256i {
        use std::arch::x86_64::*;
        let codes = unsafe { Self::raw_x86(at) };
        let offset = _mm256_set1_epi8((1 - 2 * K - LEAST) as i8);
        unsafe { _mm256_add_epi8(_mm256_add_epi8(codes, codes), offset) }
    }
}

impl<const ID: usize, const LEAST: i32, const GREATEST: i32>
    Planes<4, 0, Table<ID>, LEAST, GREATEST>
{
    /// The codebook shifted to start at zero, as unsigned bytes.
    #[cfg(target_arch = "x86_64")]
    const SHIFTED: [i8; 16] = {
        let mut shifted = [0i8; 16];
        let mut i = 0;
        while i < 16 {
            shifted[i] = (Table::<ID>::VALUES[i] as i32 - LEAST) as u8 as i8;
            i += 1;
        }
        shifted
    };
}

impl<const ID: usize, const LEAST: i32, const GREATEST: i32> Codes
    for Planes<4, 0, Table<ID>, LEAST, GREATEST>
{
    const LEAST: i32 = Self::DOMAIN.0;
    const GREATEST: i32 = Self::DOMAIN.1;
    const PANEL: Option<PanelCodes> = Some(PanelCodes { low: 4, high: 0, table: Some(Table::<ID>::VALUES), offset: 0 });

    #[inline(always)]
    unsafe fn values(at: At) -> [i8; PACKET] {
        let raw = unsafe { Self::raw(at) };
        let mut values = [0i8; PACKET];
        for i in 0..PACKET {
            values[i] = Table::<ID>::value::<4>(u32::from(raw[i])) as i8;
        }
        values
    }

    #[cfg(target_arch = "aarch64")]
    #[inline(always)]
    unsafe fn neon(at: At) -> [std::arch::aarch64::int8x16_t; 2] {
        use std::arch::aarch64::*;
        let table = unsafe { vld1q_s8(Table::<ID>::VALUES.as_ptr()) };
        unsafe { Self::raw_neon(at) }
            .map(|codes| unsafe { vqtbl1q_s8(table, vreinterpretq_u8_s8(codes)) })
    }

    #[cfg(target_arch = "aarch64")]
    #[inline(always)]
    unsafe fn neon_pair(at: At) -> [[std::arch::aarch64::int8x16_t; 2]; 2] {
        use std::arch::aarch64::*;
        let table = unsafe { vld1q_s8(Table::<ID>::VALUES.as_ptr()) };
        unsafe { Self::raw_neon_pair(at) }
            .map(|packet| packet.map(|codes| unsafe { vqtbl1q_s8(table, vreinterpretq_u8_s8(codes)) }))
    }

    #[cfg(target_arch = "aarch64")]
    #[inline(always)]
    unsafe fn neon_partners(g: At, partner: At) -> [[[std::arch::aarch64::int8x16_t; 2]; 2]; 2] {
        use std::arch::aarch64::*;
        let table = unsafe { vld1q_s8(Table::<ID>::VALUES.as_ptr()) };
        let mut codes = unsafe { Self::raw_neon_partners(g, partner) };
        for row in &mut codes {
            for packet in row {
                for half in packet {
                    *half = unsafe { vqtbl1q_s8(table, vreinterpretq_u8_s8(*half)) };
                }
            }
        }
        codes
    }

    #[cfg(target_arch = "x86_64")]
    #[inline(always)]
    unsafe fn x86(at: At) -> std::arch::x86_64::__m256i {
        unsafe { packed_x86::lookup(packed_x86::table(&Table::<ID>::VALUES), Self::raw_x86(at)) }
    }

    #[cfg(target_arch = "x86_64")]
    #[inline(always)]
    unsafe fn x86_pair(at: At) -> [std::arch::x86_64::__m256i; 2] {
        let table = unsafe { packed_x86::table(&Table::<ID>::VALUES) };
        let [first, second] = unsafe { Self::raw_x86_pair(at) };
        unsafe { [packed_x86::lookup(table, first), packed_x86::lookup(table, second)] }
    }

    #[cfg(target_arch = "x86_64")]
    #[inline(always)]
    unsafe fn x86_partners(g: At, partner: At) -> [[std::arch::x86_64::__m256i; 2]; 2] {
        let table = unsafe { packed_x86::table(&Table::<ID>::VALUES) };
        let [[a, b], [c, d]] = unsafe { Self::raw_x86_partners(g, partner) };
        unsafe {
            [
                [packed_x86::lookup(table, a), packed_x86::lookup(table, b)],
                [packed_x86::lookup(table, c), packed_x86::lookup(table, d)],
            ]
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[inline(always)]
    unsafe fn x86_shifted(at: At) -> std::arch::x86_64::__m256i {
        unsafe { packed_x86::lookup(packed_x86::table(&Self::SHIFTED), Self::raw_x86(at)) }
    }
}

impl<const LEAST: i32, const GREATEST: i32> Codes for Planes<8, 0, Twos, LEAST, GREATEST> {
    const LEAST: i32 = Self::DOMAIN.0;
    const GREATEST: i32 = Self::DOMAIN.1;
    const PANEL: Option<PanelCodes> = PanelCodes::offset(8, 0, 0);

    #[inline(always)]
    unsafe fn values(at: At) -> [i8; PACKET] {
        match at.code(RowPlaneRole::Codes) {
            CodeBytes::Packet(bytes) => unsafe { read(bytes) },
            CodeBytes::Lanes(lanes) => unsafe { lane_codes(lanes, 8) }.map(|code| code as i8),
            CodeBytes::Panel(units) => unsafe { panel_codes(units, 8) }.map(|code| code as i8),
        }
    }

    #[cfg(target_arch = "aarch64")]
    #[inline(always)]
    unsafe fn neon(at: At) -> [std::arch::aarch64::int8x16_t; 2] {
        use std::arch::aarch64::*;
        match at.code(RowPlaneRole::Codes) {
            CodeBytes::Packet(bytes) => unsafe {
                let bytes = bytes.cast::<i8>();
                [vld1q_s8(bytes), vld1q_s8(bytes.add(16))]
            },
            CodeBytes::Lanes(lanes) => unsafe { packed_neon::lane_bytes(lanes) },
            CodeBytes::Panel(units) => unsafe { packed_neon::panel::<8>(units) },
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[inline(always)]
    unsafe fn x86(at: At) -> std::arch::x86_64::__m256i {
        match at.code(RowPlaneRole::Codes) {
            CodeBytes::Packet(bytes) => unsafe { packed_x86::load(bytes.cast()) },
            CodeBytes::Lanes(lanes) => unsafe { packed_x86::lane_bytes(lanes) },
            CodeBytes::Panel(units) => unsafe { packed_x86::panel::<8>(units) },
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[inline(always)]
    unsafe fn x86_shifted(at: At) -> std::arch::x86_64::__m256i {
        use std::arch::x86_64::*;
        unsafe { _mm256_xor_si256(Self::x86(at), _mm256_set1_epi8(i8::MIN)) }
    }
}

/// Grid codes (`registry::GridCodes`): entries of grid `GRID` (its position
/// in `tables::GRIDS`), one per eight values, their `BYTES` whole bytes in
/// the codes plane and their `HIGH` further bits in the high plane
/// ([`grid_entry`]). An entry holds `8 / lanes`
/// `INDEX_BITS`-bit point indices, first lowest, then the sign bits of
/// `registry::GridSigns::ALL[SIGNS]`, then delta `D`'s bit. The value at
/// lane `j` of an entry is byte `j % lanes` of its point `j / lanes`,
/// negated by lane sign `j` (a two's complement value where the entry has no
/// signs), then delta `D`'s; the values are the registry's code domain
/// `LEAST..=GREATEST` (`DecodeAxes::codes`).
#[derive(Clone, Copy, Debug)]
pub struct Grid<
    const GRID: usize,
    const INDEX_BITS: u32,
    const BYTES: usize,
    const HIGH: u32,
    const SIGNS: usize,
    D,
    const LEAST: i32,
    const GREATEST: i32,
>(PhantomData<D>);

/// A grid entry's delta (`registry::GridCodes::delta`): its bits and the
/// values it makes of the signed values of an entry.
pub trait GridDelta: Copy + Send + Sync + 'static {
    const BITS: u32;

    /// The values of an entry of signed values `values` and delta bits
    /// `delta`.
    fn apply(values: [i8; 8], delta: u32) -> [i8; 8];
}

/// No delta: the signed values themselves.
#[derive(Clone, Copy, Debug)]
pub struct NoDelta;

impl GridDelta for NoDelta {
    const BITS: u32 = 0;

    #[inline(always)]
    fn apply(values: [i8; 8], _: u32) -> [i8; 8] {
        values
    }
}

/// A delta bit: values `2^EXPONENT v + 1`, or `2^EXPONENT v - 1` where the
/// bit is set.
#[derive(Clone, Copy, Debug)]
pub struct Delta<const EXPONENT: u32>;

impl<const EXPONENT: u32> GridDelta for Delta<EXPONENT> {
    const BITS: u32 = 1;

    #[inline(always)]
    fn apply(values: [i8; 8], delta: u32) -> [i8; 8] {
        let offset = 1 - 2 * (delta & 1) as i8;
        values.map(|value| (value << EXPONENT) + offset)
    }
}

impl<
        const GRID: usize,
        const INDEX_BITS: u32,
        const BYTES: usize,
        const HIGH: u32,
        const SIGNS: usize,
        D: GridDelta,
        const LEAST: i32,
        const GREATEST: i32,
    > Grid<GRID, INDEX_BITS, BYTES, HIGH, SIGNS, D, LEAST, GREATEST>
{
    const POINTS: tables::GridPoints = tables::GRIDS[GRID].points;
    const SIGNED: GridSigns = GridSigns::ALL[SIGNS];
    /// Sign bits an entry holds.
    const SIGN_BITS: u32 = Self::SIGNED.bits();
    const SIGN_MASK: u32 = (1 << Self::SIGN_BITS) - 1;
    /// The code domain: signed values below 128 in magnitude
    /// (`grid_values`).
    const DOMAIN: (i32, i32) = {
        assert!(
            -LEAST <= i8::MAX as i32 && GREATEST <= i8::MAX as i32,
            "grid magnitudes fit a signed byte"
        );
        (LEAST, GREATEST)
    };

}

/// Entry `e` (of four) of the packet of `at`: its `BYTES` whole bytes in its
/// grids or records plane, then its `HIGH` further bits ([`grid_high`]) from
/// bit `8 BYTES`.
#[inline(always)]
unsafe fn grid_entry<const BYTES: usize, const HIGH: u32>(at: At, e: usize) -> u32 {
    let bytes = at.entry(e, BYTES);
    (0..BYTES).fold(0, |entry, byte| {
        entry | (u32::from(unsafe { *bytes.add(byte) }) << (8 * byte))
    }) | (unsafe { grid_high::<HIGH>(at, e) } << (8 * BYTES))
}

/// The `HIGH` further bits of entry `e` (of four) of the packet of `at`, in
/// its grid high plane (`RowPlaneContent::GridHigh`, placed as its grids
/// plane is): a packet's are whole bytes ([`packet_bytes`]), entry `e`'s at
/// their bit `HIGH e`.
#[inline(always)]
unsafe fn grid_high<const HIGH: u32>(at: At, e: usize) -> u32 {
    const {
        assert!(
            HIGH <= 8 && HIGH % 2 == 0,
            "a packet's further entry bits are whole bytes, an entry's within two"
        )
    };
    if HIGH == 0 {
        return 0;
    }
    let bit = HIGH as usize * e;
    let high = packet_bytes(at, RowPlaneRole::High).wrapping_add(bit / 8);
    let mut bits = u32::from(unsafe { *high });
    if bit % 8 + HIGH as usize > 8 {
        bits |= u32::from(unsafe { *high.add(1) }) << 8;
    }
    (bits >> (bit % 8)) & ((1 << HIGH) - 1)
}

/// The eight values one grid entry gives: `point`'s bytes (byte `j` lane
/// `j`'s), negated where the lane signs `lanes` are set. Bit `j` of the
/// signs becomes byte `j` of a mask; a magnitude is below 128, so `(m ^
/// 0xff) + 1 = -m` bytewise without carries.
#[inline(always)]
fn grid_values(point: u64, lanes: u32) -> [i8; 8] {
    const ONES: u64 = 0x0101_0101_0101_0101;
    let signs = u64::from(lanes);
    let spread = signs.wrapping_mul(ONES) & 0x8040_2010_0804_0201;
    let negative = (spread.wrapping_add(0x7f7f_7f7f_7f7f_7f7f) >> 7) & ONES;
    ((point ^ (negative * 0xff)).wrapping_add(negative))
        .to_le_bytes()
        .map(|byte| byte as i8)
}

impl<
        const GRID: usize,
        const INDEX_BITS: u32,
        const BYTES: usize,
        const HIGH: u32,
        const SIGNS: usize,
        D: GridDelta,
        const LEAST: i32,
        const GREATEST: i32,
    > Codes for Grid<GRID, INDEX_BITS, BYTES, HIGH, SIGNS, D, LEAST, GREATEST>
{
    const LEAST: i32 = Self::DOMAIN.0;
    const GREATEST: i32 = Self::DOMAIN.1;

    #[inline(always)]
    unsafe fn values(at: At) -> [i8; PACKET] {
        let mask = (1u32 << INDEX_BITS) - 1;
        let mut values = [0i8; PACKET];
        for e in 0..PACKET / 8 {
            let entry = unsafe { grid_entry::<BYTES, HIGH>(at, e) };
            // The point and the entry's bits past its indices.
            let (point, rest) = match Self::POINTS {
                tables::GridPoints::Lanes8(points) => {
                    (points[(entry & mask) as usize], entry >> INDEX_BITS)
                }
                tables::GridPoints::Lanes4(points) => (
                    u64::from(points[(entry & mask) as usize])
                        | (u64::from(points[((entry >> INDEX_BITS) & mask) as usize]) << 32),
                    entry >> (2 * INDEX_BITS),
                ),
            };
            // One expression: named temporaries here change the code
            // LLVM schedules.
            values[8 * e..8 * e + 8].copy_from_slice(&D::apply(
                grid_values(point, Self::SIGNED.lanes(rest & Self::SIGN_MASK)),
                rest >> Self::SIGN_BITS,
            ));
        }
        values
    }
}

/// The encodings of a stored scale, bias or factor
/// (`registry::ScaleEncoding`), each read as its exact value.
pub mod scales {
    #[derive(Clone, Copy, Debug)]
    pub struct F16;
    #[derive(Clone, Copy, Debug)]
    pub struct Bf16;
    #[derive(Clone, Copy, Debug)]
    pub struct F32;
    /// OCP E8M0: `2^(e - 127)`; 0xff is NaN.
    #[derive(Clone, Copy, Debug)]
    pub struct E8m0;
    /// NVIDIA unsigned E4M3 (the sign bit ignored); 0x7f is NaN.
    #[derive(Clone, Copy, Debug)]
    pub struct Ue4m3;
}

/// A stored scale's encoding.
pub trait Scale: Copy + Send + Sync + 'static {
    const BYTES: usize;
    /// The exponent of the least bit a finite value of the encoding can
    /// have: its least subnormal's.
    const LEAST: i32;

    /// The scale stored at `at`.
    ///
    /// # Safety
    /// `at` addresses `BYTES` bytes.
    unsafe fn read(at: *const u8) -> f32;

    /// The eight scales at `first`, `stride` bytes apart, one a lane, each
    /// exactly as [`Scale::read`] reads it.
    ///
    /// # Safety
    /// Each addresses `BYTES` bytes; the caller's tier has 256-bit vectors
    /// and FMA.
    #[cfg(target_arch = "x86_64")]
    #[inline(always)]
    unsafe fn read8(first: *const u8, stride: usize) -> std::arch::x86_64::__m256 {
        let scales = unsafe { read8_each::<Self>(first, stride) };
        unsafe { std::arch::x86_64::_mm256_loadu_ps(scales.as_ptr()) }
    }

    /// [`Scale::read8`]'s NEON form: lanes 0..4 and 4..8.
    ///
    /// # Safety
    /// Each addresses `BYTES` bytes.
    #[cfg(target_arch = "aarch64")]
    #[inline(always)]
    unsafe fn read8_neon(
        first: *const u8,
        stride: usize,
    ) -> [std::arch::aarch64::float32x4_t; 2] {
        use std::arch::aarch64::*;
        let scales = unsafe { read8_each::<Self>(first, stride) };
        unsafe { [vld1q_f32(scales.as_ptr()), vld1q_f32(scales.as_ptr().add(4))] }
    }
}

/// The eight scales of [`Scale::read8`], each read alone.
///
/// # Safety
/// As [`Scale::read8_neon`].
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
#[inline(always)]
unsafe fn read8_each<S: Scale>(first: *const u8, stride: usize) -> [f32; 8] {
    let mut scales = [0.0f32; 8];
    for (r, scale) in scales.iter_mut().enumerate() {
        *scale = unsafe { S::read(first.add(r * stride)) };
    }
    scales
}

impl scales::F16 {
    /// The eight stored words at `first`, `stride` bytes apart.
    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
    #[inline(always)]
    unsafe fn words8(first: *const u8, stride: usize) -> [u16; 8] {
        let mut words = [0u16; 8];
        for (r, word) in words.iter_mut().enumerate() {
            *word = unsafe { read(first.add(r * stride)) };
        }
        words
    }
}

/// Eight scales widen as one vector (`Dense::widen_*`), straight from the
/// plane where they are adjacent.
impl Scale for scales::F16 {
    const BYTES: usize = 2;
    const LEAST: i32 = -24;
    #[inline(always)]
    unsafe fn read(at: *const u8) -> f32 {
        f16_to_f32(unsafe { read(at) })
    }

    #[cfg(target_arch = "x86_64")]
    #[inline(always)]
    unsafe fn read8(first: *const u8, stride: usize) -> std::arch::x86_64::__m256 {
        type Widen = crate::element::F16;
        if stride == Self::BYTES {
            return unsafe { Widen::widen_x86(first.cast()) };
        }
        let words = unsafe { Self::words8(first, stride) };
        unsafe { Widen::widen_x86(words.as_ptr()) }
    }

    #[cfg(target_arch = "aarch64")]
    #[inline(always)]
    unsafe fn read8_neon(
        first: *const u8,
        stride: usize,
    ) -> [std::arch::aarch64::float32x4_t; 2] {
        type Widen = crate::element::F16;
        if stride == Self::BYTES {
            return unsafe { Widen::widen_neon(first.cast()) };
        }
        let words = unsafe { Self::words8(first, stride) };
        unsafe { Widen::widen_neon(words.as_ptr()) }
    }
}

impl Scale for scales::Bf16 {
    const BYTES: usize = 2;
    const LEAST: i32 = -133;
    #[inline(always)]
    unsafe fn read(at: *const u8) -> f32 {
        bf16_to_f32(unsafe { read(at) })
    }
}

impl Scale for scales::F32 {
    const BYTES: usize = 4;
    const LEAST: i32 = -149;
    #[inline(always)]
    unsafe fn read(at: *const u8) -> f32 {
        f32::from_bits(unsafe { read(at) })
    }
}

impl Scale for scales::E8m0 {
    const BYTES: usize = 1;
    const LEAST: i32 = -127;
    #[inline(always)]
    unsafe fn read(at: *const u8) -> f32 {
        match unsafe { *at } {
            0 => f32::from_bits(0x0040_0000),
            0xff => f32::NAN,
            exponent => f32::from_bits(u32::from(exponent) << 23),
        }
    }
}

impl Scale for scales::Ue4m3 {
    const BYTES: usize = 1;
    const LEAST: i32 = -9;
    #[inline(always)]
    unsafe fn read(at: *const u8) -> f32 {
        let raw = unsafe { *at } & 0x7f;
        let (exponent, mantissa) = (u32::from(raw >> 3), u32::from(raw & 7));
        if raw == 0x7f {
            f32::NAN
        } else if exponent == 0 {
            // mantissa * 2^-9.
            mantissa as f32 * f32::from_bits(0x3b00_0000)
        } else {
            f32::from_bits(((exponent + 120) << 23) | (mantissa << 20))
        }
    }
}

/// `2^exponent`, exactly.
const fn power2(exponent: i32) -> f32 {
    assert!(-126 <= exponent && exponent <= 127, "a normal power of two");
    f32::from_bits(((127 + exponent) as u32) << 23)
}

/// `2^exponent` to fold into every value of scale encoding `S`, which it
/// scales exactly: none of their bits falls below `f32`'s least, `2^-149`.
/// A folded scale times a code is then one rounding of the scale times the
/// code's value, as the registry decodes it; a fold that could round fails
/// to compile.
const fn fold<S: Scale>(exponent: i32) -> f32 {
    assert!(
        exponent <= 0 && S::LEAST + exponent >= -149,
        "a power of two folds into a scale only where it scales every value exactly"
    );
    power2(exponent)
}

/// A packet's coefficients: a coefficient component.
pub trait Coefficients: Copy + Send + Sync + 'static {
    /// Whether a value is `scale * value + bias` (fused) rather than
    /// `scale * value`.
    const BIASED: bool;

    /// The (scale, bias) of values `0..16` and `16..32` of packet `at` of
    /// rows of `STORAGE`-value storage groups.
    ///
    /// # Safety
    /// `at` is a packet of a row.
    unsafe fn decode<const STORAGE: usize>(at: At) -> [(f32, f32); 2];

    /// [`Format::dot_q8`] of rows whose codes are `C`, over the first
    /// `packets` packets of block `b`.
    ///
    /// # Safety
    /// As [`Format::dot_q8`], with `packets` the block's packets below `k`.
    unsafe fn dot_q8<C: Codes, const STORAGE: usize, const R: usize, const M: usize, V: Vectors>(
        rows: &[Row; R],
        b: usize,
        packets: usize,
        x: [&Q8Block; M],
    ) -> [[f32; M]; R];

    /// Values of the units the tile kernel sums apart (`tile`): a
    /// coefficient group, at most a packet.
    const UNIT: usize;

    /// Whether each unit's coefficient is an integer (a scale code), so the
    /// units' widened sums add exactly into one per block (`tile::Scaled`).
    const INTEGER_UNITS: bool;

    /// The largest magnitude of a unit's integer and of a minimum code.
    const MAGNITUDE: i32;

    /// The tile kernel's exact integers of a block against `W` activation
    /// rows: `tile::Scaled` with integer unit coefficients, else
    /// `tile::Sums`.
    type Units<const W: usize>: tile::Integers<W>;

    /// The coefficients of block `b` (its first `packets` packets) of the
    /// eight `rows` as the tile kernel's combine takes them: with `panel`
    /// (lanes of one rows8 tile, `Row::panel_tile`) each record's eight
    /// read as vectors from its span, else row by row.
    ///
    /// # Safety
    /// Each row is a row of the format, and block `b` lies in it.
    unsafe fn tile_side<const STORAGE: usize, V: Vectors>(rows: &[Row; 8], b: usize, packets: usize, panel: bool, side: &mut tile::Side);

    /// Each of the `W` activation rows' products with the eight rows, from
    /// the core's exact integers `units`: the single-row dot's `f32`
    /// arithmetic, in its order, every row a lane.
    fn tile_combine<const W: usize, const R: usize>(side: &tile::Side, units: &Self::Units<W>, x: &Q8Tile<R>) -> [[f32; 8]; W];

    /// [`Format::dot_q8_tiles`] of rows whose codes are `C`.
    ///
    /// # Safety
    /// As [`Format::dot_q8_tiles`].
    #[inline(always)]
    unsafe fn dot_q8_tiles<C: Codes, V: Vectors>(first: *const u8, geometry: &RowGeometry, x: &[Q8Block], k: usize, out: &mut [f32]) -> bool {
        let _ = (first, geometry, x, k, out);
        false
    }

    /// The quantized dot's contract for block `b` (its first `packets`
    /// packets) of `row` and `x`, written plainly from the row's values and
    /// coefficients as the decode reads them: the tests' reference.
    #[cfg(test)]
    unsafe fn reference<C: Codes, const STORAGE: usize>(row: &Row, b: usize, packets: usize, x: &Q8Block) -> f32;
}

/// The activation code sum of values `16 h .. 16 h + 16` of packet `local`
/// of `x`'s block, or with `whole` of the whole packet.
#[inline(always)]
fn activation_sum(x: &Q8Block, local: usize, half: usize, whole: bool) -> i32 {
    if whole {
        i32::from(x.sums[2 * local]) + i32::from(x.sums[2 * local + 1])
    } else {
        i32::from(x.sums[2 * local + half])
    }
}

/// The sum of the eight i32 lanes of `lanes`.
#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn sum_lanes(lanes: std::arch::x86_64::__m256i) -> i32 {
    use std::arch::x86_64::*;
    unsafe {
        let half = _mm_add_epi32(_mm256_castsi256_si128(lanes), _mm256_extracti128_si256::<1>(lanes));
        let pair = _mm_add_epi32(half, _mm_shuffle_epi32::<0b01_00_11_10>(half));
        _mm_cvtsi128_si32(_mm_add_epi32(pair, _mm_shuffle_epi32::<0b10_11_00_01>(pair)))
    }
}

/// A k-block's 64 activation codes at `x` in the byte order of an `mma16`
/// lane pair's code bytes (`Hier::dot_tile16`): register `p` holds lanes
/// `2p` and `2p + 1`, byte `16t + 4s + j` column `16s + [2t, 2t + 8, 2t +
/// 1, 2t + 9][j]`.
#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn permuted_activations(x: *const i8) -> [std::arch::x86_64::__m256i; 2] {
    use std::arch::x86_64::*;
    // The byte of step `s` of lane `t` taking column `c` of the step, or
    // none (0x80).
    const fn masks() -> [[[i8; 32]; 4]; 2] {
        let mut masks = [[[-128i8; 32]; 4]; 2];
        let mut pair = 0;
        while pair < 2 {
            let mut s = 0;
            while s < 4 {
                let mut half = 0;
                while half < 2 {
                    let t = 2 * pair + half;
                    let columns = [2 * t, 2 * t + 8, 2 * t + 1, 2 * t + 9];
                    let mut j = 0;
                    while j < 4 {
                        masks[pair][s][16 * half + 4 * s + j] = columns[j] as i8;
                        j += 1;
                    }
                    half += 1;
                }
                s += 1;
            }
            pair += 1;
        }
        masks
    }
    const MASKS: [[[i8; 32]; 4]; 2] = masks();
    // Loops, not closures: a closure would not take the caller's target
    // features, and its intrinsics would not inline.
    let mut out = [unsafe { _mm256_setzero_si256() }; 2];
    for s in 0..4 {
        unsafe {
            let step = _mm256_broadcastsi128_si256(_mm_loadu_si128(x.add(16 * s).cast()));
            for pair in 0..2 {
                out[pair] = _mm256_or_si256(out[pair], _mm256_shuffle_epi8(step, _mm256_loadu_si256(MASKS[pair][s].as_ptr().cast())));
            }
        }
    }
    out
}

/// The high bits of an `mma16` lane pair's codes at `bytes` (`high` bits a
/// code: a lane's step of `high` bytes, slot `q` at bit `high q`), shifted
/// to bit 4, in the byte order of its low nibbles (`permuted_activations`'):
/// row `g`'s (slot `2j` of byte `j`'s step), then row `g + 8`'s (slot
/// `2j + 1`).
#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn high_bits(bytes: *const u8, high: u32) -> [std::arch::x86_64::__m256i; 2] {
    use std::arch::x86_64::*;
    unsafe {
        match high {
            1 => {
                // Lane t's step s is byte 4t + s; slot q is bit q.
                const SPREAD: [i8; 32] = {
                    let mut spread = [0i8; 32];
                    let mut i = 0;
                    while i < 32 {
                        spread[i] = (4 * (i / 16) + (i % 16) / 4) as i8;
                        i += 1;
                    }
                    spread
                };
                let lanes = _mm256_set1_epi64x(std::ptr::read_unaligned(bytes.cast::<i64>()));
                let spread = _mm256_shuffle_epi8(lanes, _mm256_loadu_si256(SPREAD.as_ptr().cast()));
                // Byte j of a step: bits 2j (row g) and 2j + 1 (row g + 8).
                let mut rows = [_mm256_setzero_si256(); 2];
                for (upper, first) in [1i32, 2].into_iter().enumerate() {
                    let bits = _mm256_set1_epi32(first | first << 10 | first << 20 | first << 30);
                    let set = _mm256_cmpeq_epi8(_mm256_and_si256(spread, bits), bits);
                    rows[upper] = _mm256_and_si256(set, _mm256_set1_epi8(0x10));
                }
                rows
            }
            2 => {
                // Lane t's step s is bytes 8t + 2s, +1; slot q is bits 2q,
                // byte q / 4 of the step: byte j's slots 2j, 2j + 1 lie in
                // the step's byte j / 2, at bits 4 (j % 2) and 4 (j % 2) + 2.
                const SPREAD: [i8; 32] = {
                    let mut spread = [0i8; 32];
                    let mut i = 0;
                    while i < 32 {
                        let (t, s, j) = (i / 16, (i % 16) / 4, i % 4);
                        spread[i] = (8 * t + 2 * s + j / 2) as i8;
                        i += 1;
                    }
                    spread
                };
                const ODD: [i8; 32] = {
                    let mut odd = [0i8; 32];
                    let mut i = 0;
                    while i < 32 {
                        odd[i] = if i % 2 == 1 { -1 } else { 0 };
                        i += 1;
                    }
                    odd
                };
                let lanes = _mm256_broadcastsi128_si256(_mm_loadu_si128(bytes.cast()));
                let spread = _mm256_shuffle_epi8(lanes, _mm256_loadu_si256(SPREAD.as_ptr().cast()));
                // Odd bytes j take the step byte's upper nibble.
                let nibbles = _mm256_blendv_epi8(spread, _mm256_srli_epi16::<4>(spread), _mm256_loadu_si256(ODD.as_ptr().cast()));
                let two = _mm256_set1_epi8(0x03);
                let first = _mm256_and_si256(nibbles, two);
                let second = _mm256_and_si256(_mm256_srli_epi16::<2>(nibbles), two);
                [_mm256_slli_epi16::<4>(first), _mm256_slli_epi16::<4>(second)]
            }
            _ => unreachable!("a high plane of at most 2 bits"),
        }
    }
}

/// Coefficient groups of a packet under `GROUP`-value groups: 1 or 2.
const fn groups_per_packet(group: usize) -> usize {
    assert!(
        group == 16 || group % PACKET == 0,
        "coefficient groups of 16 or whole packets"
    );
    if group == 16 {
        2
    } else {
        1
    }
}

/// The exact integer dots of the first `packets` packets of `block` with
/// each activation block of `x`: `[m][h][local]`, per `GROUP`-value
/// coefficient group of a packet (the whole packet in `h = 0` when a group
/// covers it), as `f32`: a dot of 32 products of at most 128 x 255 is below
/// 2^24 in magnitude, so its `f32` is exact, and it converts in the vectors
/// that reduce it.
#[inline(always)]
unsafe fn packet_dots<
    C: Codes,
    const STORAGE: usize,
    const M: usize,
    const GROUP: usize,
    V: Vectors,
>(
    block: &Block,
    packets: usize,
    x: [&Q8Block; M],
) -> [[[f32; 8]; 2]; M] {
    let halves = const { groups_per_packet(GROUP) };
    let mut exact = [[[0.0f32; 8]; 2]; M];
    #[cfg(target_arch = "aarch64")]
    if V::QUANTIZED_DOTS >= 1 {
        use std::arch::aarch64::*;
        // A packet's products are four lanes; pairwise adds reduce four
        // packets' into one vector of their sums, which stays a vector (no
        // lane-by-lane reduction into general registers).
        let zero = unsafe { vdupq_n_s32(0) };
        let mut unpaired = [[zero; 2]; M];
        let mut pairs = [[[zero; 4]; 2]; M];
        for local in packets_of(packets) {
            let weights = unsafe { C::neon(block.at::<STORAGE>(local)) };
            for m in 0..M {
                let activations = x[m].codes[PACKET * local..].as_ptr();
                let (first, second) =
                    unsafe { (vld1q_s8(activations), vld1q_s8(activations.add(16))) };
                let products = unsafe {
                    if halves == 2 {
                        [
                            packed_neon::sdot(zero, weights[0], first),
                            packed_neon::sdot(zero, weights[1], second),
                        ]
                    } else {
                        let products = packed_neon::sdot(
                            packed_neon::sdot(zero, weights[0], first),
                            weights[1],
                            second,
                        );
                        [products, zero]
                    }
                };
                for h in 0..halves {
                    if local % 2 == 0 {
                        unpaired[m][h] = products[h];
                    } else {
                        pairs[m][h][local / 2] = unsafe { vpaddq_s32(unpaired[m][h], products[h]) };
                    }
                }
            }
        }
        for m in 0..M {
            for h in 0..halves {
                if packets % 2 == 1 {
                    pairs[m][h][packets / 2] = unsafe { vpaddq_s32(unpaired[m][h], zero) };
                }
                let pairs = pairs[m][h];
                unsafe {
                    let quads = [vpaddq_s32(pairs[0], pairs[1]), vpaddq_s32(pairs[2], pairs[3])];
                    vst1q_f32(exact[m][h].as_mut_ptr(), vcvtq_f32_s32(quads[0]));
                    vst1q_f32(exact[m][h].as_mut_ptr().add(4), vcvtq_f32_s32(quads[1]));
                }
            }
        }
        return exact;
    }
    #[cfg(target_arch = "x86_64")]
    if V::QUANTIZED_DOTS >= 1 {
        unsafe {
            use std::arch::x86_64::*;
            let shift = const { packed_x86::shift::<C>(packed_x86::Products::of::<V>()) };
            let least = const {
                match packed_x86::shift::<C>(packed_x86::Products::of::<V>()) {
                    Some(least) => least,
                    None => 0,
                }
            };
            // Per activation block, a block's eight packets reduce together:
            // the sums of their low and high halves (values 0..16, 16..32).
            for m in 0..M {
                let mut products = [_mm256_setzero_si256(); 8];
                for local in packets_of(packets) {
                    let weights = C::x86(block.at::<STORAGE>(local));
                    products[local] = packed_x86::product::<V>(weights, x[m], local, shift);
                }
                let (mut first, mut second) = packed_x86::halves8(products);
                if least != 0 {
                    // The shift's correction, `least Σ a`, a packet a lane:
                    // its halves' block sums times `least` (and 0), paired.
                    let sums = _mm256_loadu_si256(x[m].sums.as_ptr().cast());
                    let (low, high) = (least & 0xffff, least << 16);
                    first = _mm256_add_epi32(first, _mm256_madd_epi16(sums, _mm256_set1_epi32(low)));
                    second = _mm256_add_epi32(second, _mm256_madd_epi16(sums, _mm256_set1_epi32(high)));
                }
                if halves == 1 {
                    let whole = _mm256_add_epi32(first, second);
                    _mm256_storeu_ps(exact[m][0].as_mut_ptr(), _mm256_cvtepi32_ps(whole));
                } else {
                    _mm256_storeu_ps(exact[m][0].as_mut_ptr(), _mm256_cvtepi32_ps(first));
                    _mm256_storeu_ps(exact[m][1].as_mut_ptr(), _mm256_cvtepi32_ps(second));
                }
            }
        }
        return exact;
    }
    #[allow(unreachable_code)]
    {
        for local in packets_of(packets) {
            let values = unsafe { C::values(block.at::<STORAGE>(local)) };
            let dots = scalar_dots::<M, GROUP, V>(&values, x, local);
            for m in 0..M {
                for h in 0..halves {
                    exact[m][h][local] = dots[m][h] as f32;
                }
            }
        }
        exact
    }
}

/// The exact integer dots of one packet's `values` with packet `local` of
/// each activation block of `x`, per `GROUP`-value coefficient group (the
/// whole packet in `[0]` when a group covers it), by [`dot_i8_i8`].
#[inline(always)]
fn scalar_dots<const M: usize, const GROUP: usize, V: Vectors>(
    values: &[i8; PACKET],
    x: [&Q8Block; M],
    local: usize,
) -> [[i32; 2]; M] {
    let halves = const { groups_per_packet(GROUP) };
    let mut exact = [[0i32; 2]; M];
    for (exact, x) in exact.iter_mut().zip(x.iter()) {
        let activations: &[i8; PACKET] = x.codes[PACKET * local..PACKET * (local + 1)]
            .try_into()
            .expect("a packet of codes");
        for (h, exact) in exact.iter_mut().enumerate().take(halves) {
            *exact = if halves == 1 {
                dot_i8_i8::<V>(values, activations)
            } else {
                // Each sixteen-value group alone: the other half is zero.
                let mut half = [0i8; PACKET];
                half[16 * h..16 * h + 16].copy_from_slice(&values[16 * h..16 * h + 16]);
                dot_i8_i8::<V>(&half, activations)
            };
        }
    }
    exact
}

/// Direct coefficients (`CodingAxes::Direct`): per `GROUP` values a
/// stored scale of encoding `S` (and with `BIAS` a bias), the scales of a
/// storage group then its biases in the supers plane; values are scaled by
/// `2^CODE_EXPONENT`, folded into each scale exactly. A quantized dot sums
/// each group's exact integer products and accumulates `scale * products`
/// (and `bias * activation sum`), fused, in group order.
#[derive(Clone, Copy, Debug)]
pub struct Direct<S, const GROUP: usize, const BIAS: bool, const CODE_EXPONENT: i32>(
    PhantomData<S>,
);

impl<S: Scale, const GROUP: usize, const BIAS: bool, const CODE_EXPONENT: i32>
    Direct<S, GROUP, BIAS, CODE_EXPONENT>
{
    /// [`Coefficients::dot_q8`] of rows against four activation blocks on
    /// x86: per packet the four blocks' exact sums of a group in one vector,
    /// a block a lane, and each lane's combine a fused multiply-add in group
    /// order, as a block's own.
    #[cfg(target_arch = "x86_64")]
    #[inline(always)]
    unsafe fn dot_q8_four<
        C: Codes,
        const STORAGE: usize,
        const R: usize,
        const M: usize,
        V: Vectors,
    >(
        rows: &[Row; R],
        b: usize,
        packets: usize,
        x: [&Q8Block; M],
    ) -> [[f32; M]; R] {
        use std::arch::x86_64::*;
        let halves = const { groups_per_packet(GROUP) };
        let shift = const { packed_x86::shift::<C>(packed_x86::Products::of::<V>()) };
        let least = const {
            match packed_x86::shift::<C>(packed_x86::Products::of::<V>()) {
                Some(least) => least,
                None => 0,
            }
        };
        let mut out = [[0.0f32; M]; R];
        for (out, row) in out.iter_mut().zip(rows) {
            let block = Block::of::<STORAGE>(row, b);
            let mut sums = unsafe { _mm_setzero_ps() };
            for local in packets_of(packets) {
                let at = block.at::<STORAGE>(local);
                let weights = unsafe { C::x86(at) };
                let mut products = [unsafe { _mm256_setzero_si256() }; 4];
                for (products, x) in products.iter_mut().zip(x.iter()) {
                    *products = unsafe { packed_x86::product::<V>(weights, x, local, shift) };
                }
                // Low and high halves of the four blocks' sums.
                let halves4 = unsafe { packed_x86::halves4(products) };
                let (low, high) = unsafe {
                    (
                        _mm256_castsi256_si128(halves4),
                        _mm256_extracti128_si256::<1>(halves4),
                    )
                };
                let exact = if halves == 1 {
                    [unsafe { _mm_add_epi32(low, high) }, low]
                } else {
                    [low, high]
                };
                let coefficients = unsafe { Self::decode::<STORAGE>(at) };
                for h in 0..halves {
                    let (exact, (scale, bias)) = (exact[h], coefficients[h]);
                    let mut codes = [0i32; 4];
                    for m in 0..4 {
                        codes[m] = activation_sum(x[m], local, h, halves == 1);
                    }
                    let codes = unsafe { _mm_loadu_si128(codes.as_ptr().cast()) };
                    let exact = if least != 0 { unsafe { _mm_add_epi32(exact, _mm_mullo_epi32(_mm_set1_epi32(least), codes)) } } else { exact };
                    sums =
                        unsafe { _mm_fmadd_ps(_mm_set1_ps(scale), _mm_cvtepi32_ps(exact), sums) };
                    if BIAS {
                        sums = unsafe {
                            _mm_fmadd_ps(_mm_set1_ps(bias), _mm_cvtepi32_ps(codes), sums)
                        };
                    }
                }
            }
            let mut lanes = [0.0f32; 4];
            unsafe { _mm_storeu_ps(lanes.as_mut_ptr(), sums) };
            for ((out, lane), x) in out.iter_mut().zip(lanes).zip(x.iter()) {
                *out = x.d * lane;
            }
        }
        out
    }

    /// [`Self::dot_q8_four`]'s NEON form: per packet the four blocks' exact
    /// sums of a group in one vector, a block a lane, and each lane's
    /// combine a fused multiply-add in group order, as a block's own.
    #[cfg(target_arch = "aarch64")]
    #[inline(always)]
    unsafe fn dot_q8_four_neon<
        C: Codes,
        const STORAGE: usize,
        const R: usize,
        const M: usize,
        V: Vectors,
    >(
        rows: &[Row; R],
        b: usize,
        packets: usize,
        x: [&Q8Block; M],
    ) -> [[f32; M]; R] {
        use std::arch::aarch64::*;
        let halves = const { groups_per_packet(GROUP) };
        let zero = unsafe { vdupq_n_s32(0) };
        let mut out = [[0.0f32; M]; R];
        for (out, row) in out.iter_mut().zip(rows) {
            let block = Block::of::<STORAGE>(row, b);
            let mut sums = unsafe { vdupq_n_f32(0.0) };
            for local in packets_of(packets) {
                let at = block.at::<STORAGE>(local);
                let weights = unsafe { C::neon(at) };
                let mut products = [[zero; 4]; 2];
                for m in 0..4 {
                    let activations = x[m].codes[PACKET * local..].as_ptr();
                    unsafe {
                        let (first, second) =
                            (vld1q_s8(activations), vld1q_s8(activations.add(16)));
                        if halves == 1 {
                            products[0][m] = packed_neon::sdot(
                                packed_neon::sdot(zero, weights[0], first),
                                weights[1],
                                second,
                            );
                        } else {
                            products[0][m] = packed_neon::sdot(zero, weights[0], first);
                            products[1][m] = packed_neon::sdot(zero, weights[1], second);
                        }
                    }
                }
                let coefficients = unsafe { Self::decode::<STORAGE>(at) };
                for (h, products) in products.iter().enumerate().take(halves) {
                    let exact = unsafe {
                        vcvtq_f32_s32(vpaddq_s32(
                            vpaddq_s32(products[0], products[1]),
                            vpaddq_s32(products[2], products[3]),
                        ))
                    };
                    let (scale, bias) = coefficients[h];
                    sums = unsafe { vfmaq_n_f32(sums, exact, scale) };
                    if BIAS {
                        let mut codes = [0.0f32; 4];
                        for m in 0..4 {
                            codes[m] = activation_sum(x[m], local, h, halves == 1) as f32;
                        }
                        sums = unsafe { vfmaq_n_f32(sums, vld1q_f32(codes.as_ptr()), bias) };
                    }
                }
            }
            let mut lanes = [0.0f32; 4];
            unsafe { vst1q_f32(lanes.as_mut_ptr(), sums) };
            for ((out, lane), x) in out.iter_mut().zip(lanes).zip(x.iter()) {
                *out = x.d * lane;
            }
        }
        out
    }

    /// The (scale, bias) of coefficient group `local` of the storage group
    /// of `at`: supers records `local`, and the bias's past every scale.
    #[inline(always)]
    unsafe fn group<const STORAGE: usize>(at: At, local: usize) -> (f32, f32) {
        let field = |record| at.field(RowPlaneRole::Supers, record, 8 * S::BYTES).0;
        let scale = unsafe { S::read(field(local)) };
        let scale = if CODE_EXPONENT == 0 { scale } else { scale * const { fold::<S>(CODE_EXPONENT) } };
        let bias = if BIAS {
            unsafe { S::read(field(STORAGE / GROUP + local)) }
        } else {
            0.0
        };
        (scale, bias)
    }
}

impl<S: Scale, const GROUP: usize, const BIAS: bool, const CODE_EXPONENT: i32> Coefficients
    for Direct<S, GROUP, BIAS, CODE_EXPONENT>
{
    const BIASED: bool = BIAS;
    const UNIT: usize = if GROUP < PACKET { GROUP } else { PACKET };
    const INTEGER_UNITS: bool = false;
    const MAGNITUDE: i32 = 1;
    type Units<const W: usize> = tile::Sums<W>;

    #[inline(always)]
    unsafe fn tile_side<const STORAGE: usize, V: Vectors>(rows: &[Row; 8], b: usize, packets: usize, panel: bool, side: &mut tile::Side) {
        if panel {
            // Each record's eight rows together in its span (`At::field`).
            let block = Block::of::<STORAGE>(&rows[0], b);
            for local in packets_of(packets) {
                let at = block.at::<STORAGE>(local);
                for h in 0..PACKET / Self::UNIT {
                    let (u, record) = (local * PACKET / Self::UNIT + h, at.local::<GROUP>(h));
                    let field = |record| at.field(RowPlaneRole::Supers, record, 8 * S::BYTES).0;
                    let scales = unsafe { tile::read8::<S, V>(field(record)) };
                    // A loop, not an array `map`: it would compile out of line.
                    for (side, scale) in side.scales[u].iter_mut().zip(scales) {
                        *side = if CODE_EXPONENT == 0 { scale } else { scale * const { fold::<S>(CODE_EXPONENT) } };
                    }
                    if BIAS {
                        side.biases[u] = unsafe { tile::read8::<S, V>(field(STORAGE / GROUP + record)) };
                    }
                }
            }
            return;
        }
        for (r, row) in rows.iter().enumerate() {
            let block = Block::of::<STORAGE>(row, b);
            for local in packets_of(packets) {
                let coefficients = unsafe { Self::decode::<STORAGE>(block.at::<STORAGE>(local)) };
                for (h, (scale, bias)) in coefficients.into_iter().enumerate().take(PACKET / Self::UNIT) {
                    let u = local * PACKET / Self::UNIT + h;
                    (side.scales[u][r], side.biases[u][r]) = (scale, bias);
                }
            }
        }
    }

    #[inline(always)]
    fn tile_combine<const W: usize, const R: usize>(side: &tile::Side, units: &tile::Sums<W>, x: &Q8Tile<R>) -> [[f32; 8]; W] {
        let mut out = [[0.0f32; 8]; W];
        for (a, out) in out.iter_mut().enumerate() {
            // Each row's fused sum in group order, as `Coefficients::dot_q8`.
            let mut sums = [0.0f32; 8];
            for u in 0..units.count {
                let codes = if BIAS { tile::unit_sum(x, Self::UNIT, u, a) } else { 0 };
                for r in 0..8 {
                    sums[r] = side.scales[u][r].mul_add(units.units[u][a][r] as f32, sums[r]);
                    if BIAS {
                        sums[r] = side.biases[u][r].mul_add(codes as f32, sums[r]);
                    }
                }
            }
            for (out, sum) in out.iter_mut().zip(sums) {
                *out = x.d[a] * sum;
            }
        }
        out
    }

    #[cfg(test)]
    unsafe fn reference<C: Codes, const STORAGE: usize>(row: &Row, b: usize, packets: usize, x: &Q8Block) -> f32 {
        // Per unit in group order: `fma(scale, Σ v a, sum)`, then
        // `fma(bias, Σ a, sum)`; then `d * sum`.
        let (block, mut sum) = (Block::of::<STORAGE>(row, b), 0.0f32);
        for local in packets_of(packets) {
            let at = block.at::<STORAGE>(local);
            let (coefficients, values) = unsafe { (Self::decode::<STORAGE>(at), C::values(at)) };
            for (h, (scale, bias)) in coefficients.into_iter().enumerate().take(PACKET / Self::UNIT) {
                let unit = h * Self::UNIT..(h + 1) * Self::UNIT;
                let codes = |i: usize| i32::from(x.codes[PACKET * local + i]);
                let exact: i32 = unit.clone().map(|i| i32::from(values[i]) * codes(i)).sum();
                sum = scale.mul_add(exact as f32, sum);
                if BIAS {
                    sum = bias.mul_add(unit.map(codes).sum::<i32>() as f32, sum);
                }
            }
        }
        x.d * sum
    }

    #[inline(always)]
    unsafe fn decode<const STORAGE: usize>(at: At) -> [(f32, f32); 2] {
        let first = unsafe { Self::group::<STORAGE>(at, at.local::<GROUP>(0)) };
        if GROUP == 16 {
            [first, unsafe {
                Self::group::<STORAGE>(at, at.local::<GROUP>(1))
            }]
        } else {
            [first, first]
        }
    }

    #[inline(always)]
    unsafe fn dot_q8<C: Codes, const STORAGE: usize, const R: usize, const M: usize, V: Vectors>(
        rows: &[Row; R],
        b: usize,
        packets: usize,
        x: [&Q8Block; M],
    ) -> [[f32; M]; R] {
        let halves = const { groups_per_packet(GROUP) };
        #[cfg(target_arch = "x86_64")]
        if M == 4 && V::QUANTIZED_DOTS >= 1 {
            return unsafe { Self::dot_q8_four::<C, STORAGE, R, M, V>(rows, b, packets, x) };
        }
        #[cfg(target_arch = "aarch64")]
        if M == 4 && V::QUANTIZED_DOTS >= 1 {
            return unsafe { Self::dot_q8_four_neon::<C, STORAGE, R, M, V>(rows, b, packets, x) };
        }
        let mut out = [[0.0f32; M]; R];
        for (out, row) in out.iter_mut().zip(rows) {
            let block = Block::of::<STORAGE>(row, b);
            let exact = unsafe { packet_dots::<C, STORAGE, M, GROUP, V>(&block, packets, x) };
            let mut sums = [0.0f32; M];
            for local in packets_of(packets) {
                let coefficients = unsafe { Self::decode::<STORAGE>(block.at::<STORAGE>(local)) };
                for m in 0..M {
                    for h in 0..halves {
                        let (scale, bias) = coefficients[h];
                        sums[m] = scale.mul_add(exact[m][h][local], sums[m]);
                        if BIAS {
                            let codes = activation_sum(x[m], local, h, halves == 1);
                            sums[m] = bias.mul_add(codes as f32, sums[m]);
                        }
                    }
                }
            }
            for ((out, sum), x) in out.iter_mut().zip(sums).zip(x.iter()) {
                *out = x.d * sum;
            }
        }
        out
    }
}

/// The local codes of hierarchical coefficients: per coefficient group a
/// scale code (and with a bias its bias code).
pub trait Locals: Copy + Send + Sync + 'static {
    /// The groups' bias term, if any.
    const BIAS: Option<LocalBias>;

    /// The largest magnitude of a scale or bias code.
    const MAGNITUDE: i32;

    /// Whether [`Locals::x86_groups`] decodes the codes as vectors.
    const X86_GROUPS: bool = false;

    /// The (scale, bias) codes' values of every `GROUP`-value coefficient
    /// group of the storage group of `at`, as vectors of i32 lanes in group
    /// order (groups `0..8` in `[0]`, `8..16` in `[1]`): `[scales,
    /// biases]`, the bias codes 0 without a bias. None without a vector
    /// form ([`Locals::X86_GROUPS`]).
    ///
    /// # Safety
    /// `scales` is the storage group's scales field (`At::plane` of
    /// `RowPlaneRole::Scales`); the caller's tier has 256-bit integer
    /// vectors.
    #[cfg(target_arch = "x86_64")]
    #[inline(always)]
    unsafe fn x86_groups<const GROUP: usize>(scales: *const u8) -> Option<[[std::arch::x86_64::__m256i; 2]; 2]> {
        let _ = scales;
        None
    }

    /// The (scale, bias) codes of records `record ..` of a rows8 span (`at`
    /// a storage group of a tile's lane 0), a record of its eight rows a
    /// vector, the next eight a second with groups of 16 values: `[scales,
    /// biases]`, as [`Locals::x86_groups`] decodes them. None without a
    /// vector form on tier `V`.
    ///
    /// # Safety
    /// `at` is a storage group of lane 0 of a rows8 tile.
    #[inline(always)]
    unsafe fn groups8<const GROUP: usize, V: Vectors>(at: At, record: usize) -> Option<[[[i32; 8]; 2]; 2]> {
        let _ = (at, record);
        None
    }

    /// The (scale, bias) codes' values of `GROUP`-value coefficient group
    /// `local` of the storage group of `at`; the bias code is 0 without a
    /// bias.
    ///
    /// # Safety
    /// `at` is a packet of a row.
    unsafe fn read<const GROUP: usize>(at: At, local: usize) -> (i32, i32);

    /// The codes of every packet of the 256-value storage group of `at`,
    /// per packet and half (a group's codes in both halves of a packet it
    /// covers): what a quantized dot reads once per row and block.
    ///
    /// # Safety
    /// As [`Locals::read`].
    #[inline(always)]
    unsafe fn block<const GROUP: usize>(at: At) -> [[(i32, i32); 2]; 8] {
        let mut codes = [[(0, 0); 2]; 8];
        for (packet, codes) in codes.iter_mut().enumerate() {
            for (half, codes) in codes.iter_mut().enumerate() {
                *codes = unsafe { Self::read::<GROUP>(at, at.packet(packet).local::<GROUP>(half)) };
            }
        }
        codes
    }
}

/// Local codes in the scales plane: per coefficient group a `BITS`-bit
/// scale code (then, with bias `B`'s minimum, a minimum code) read through
/// `L`, group `j`'s at bit `j BITS (1 + minimum)` of its storage group's
/// bytes.
#[derive(Clone, Copy, Debug)]
pub struct Scales<L, const BITS: u32, B>(PhantomData<(L, B)>);

/// The bias of local codes in the scales plane (`registry::LocalBias`):
/// none, or a minimum code.
pub trait PlaneBias: Copy + Send + Sync + 'static {
    const BIAS: Option<LocalBias>;
}

/// A minimum code (`LocalBias::Minimum`), its term subtracted when
/// `SUBTRACT`, else added.
#[derive(Clone, Copy, Debug)]
pub struct Minimum<const SUBTRACT: bool>;

impl<const SUBTRACT: bool> PlaneBias for Minimum<SUBTRACT> {
    const BIAS: Option<LocalBias> = Some(LocalBias::Minimum(if SUBTRACT {
        MinimumSign::Subtract
    } else {
        MinimumSign::Add
    }));
}

impl<L: Interpretation, const BITS: u32, B: PlaneBias> Scales<L, BITS, B> {
    /// Whether the groups have a minimum code.
    const MINIMUM: bool = B::BIAS.is_some();

    /// Bits of one group's codes; every group's lie within two bytes.
    const WIDTH: usize = {
        let width = BITS as usize * (1 + Self::MINIMUM as usize);
        // Every record of a row, or of a rows8 span's eight rows.
        let mut group = 0;
        while group < 128 {
            assert!(
                (group * width) % 8 + width <= 16,
                "a group's codes lie within two bytes"
            );
            group += 1;
        }
        width
    };

    /// The (scale, minimum) codes' values of the field at bit `bit % 8` of
    /// the low bits of `bytes`.
    #[inline(always)]
    fn codes(bytes: u32, bit: usize) -> (i32, i32) {
        Self::split(bytes >> (bit % 8))
    }

    /// The (scale, minimum) codes' values of the field in the low bits of
    /// `field`.
    #[inline(always)]
    fn split(field: u32) -> (i32, i32) {
        let mask = (1u32 << BITS) - 1;
        let minimum = if Self::MINIMUM {
            L::value::<BITS>((field >> BITS) & mask)
        } else {
            0
        };
        (L::value::<BITS>(field & mask), minimum)
    }
}

impl<L: Interpretation, const BITS: u32, B: PlaneBias> Locals for Scales<L, BITS, B> {
    const BIAS: Option<LocalBias> = B::BIAS;
    // A `BITS`-bit field's value: unsigned, two's complement, or less an
    // offset within the field's range.
    const MAGNITUDE: i32 = 1 << BITS;
    // Each group's field within three bytes from its first.
    const X86_GROUPS: bool = Self::WIDTH <= 17;

    #[cfg(target_arch = "x86_64")]
    #[inline(always)]
    unsafe fn x86_groups<const GROUP: usize>(scales: *const u8) -> Option<[[std::arch::x86_64::__m256i; 2]; 2]> {
        use std::arch::x86_64::*;
        if !Self::X86_GROUPS {
            return None;
        }
        let groups = Q8_BLOCK / GROUP;
        let bytes = (groups * Self::WIDTH).div_ceil(8);
        let mut fields = [0u8; 16];
        unsafe { std::ptr::copy_nonoverlapping(scales, fields.as_mut_ptr(), bytes.min(16)) };
        // Lane `i` of vector `v` takes group `8v + i`'s field: its first
        // byte and the next two as the lane's low bytes, shifted down by its
        // first bit's place in its first byte.
        let (select, shifts) = const {
            let groups = Q8_BLOCK / GROUP;
            let mut select = [[-128i8; 32]; 2];
            let mut shifts = [[0i32; 8]; 2];
            let mut group = 0;
            while group < groups && group < 16 {
                let (vector, lane) = (group / 8, group % 8);
                let bit = group * Self::WIDTH;
                let mut byte = 0;
                while byte < 3 {
                    let at = bit / 8 + byte;
                    if at < 16 {
                        select[vector][4 * lane + byte] = at as i8;
                    }
                    byte += 1;
                }
                shifts[vector][lane] = (bit % 8) as i32;
                group += 1;
            }
            (select, shifts)
        };
        let mask = unsafe { _mm256_set1_epi32((1 << BITS) - 1) };
        let mut out = [[unsafe { _mm256_setzero_si256() }; 2]; 2];
        unsafe {
            let fields = _mm256_broadcastsi128_si256(_mm_loadu_si128(fields.as_ptr().cast()));
            for vector in 0..groups.div_ceil(8) {
                let field = _mm256_srlv_epi32(
                    _mm256_shuffle_epi8(fields, _mm256_loadu_si256(select[vector].as_ptr().cast())),
                    _mm256_loadu_si256(shifts[vector].as_ptr().cast()),
                );
                out[0][vector] = L::values8::<BITS>(_mm256_and_si256(field, mask));
                if Self::MINIMUM {
                    out[1][vector] = L::values8::<BITS>(_mm256_and_si256(_mm256_srl_epi32(field, _mm_cvtsi32_si128(BITS as i32)), mask));
                }
            }
        }
        Some(out)
    }

    #[inline(always)]
    unsafe fn groups8<const GROUP: usize, V: Vectors>(at: At, record: usize) -> Option<[[[i32; 8]; 2]; 2]> {
        // A record's eight rows' fields one after another: a storage group's
        // fields as a row of rows16 holds them.
        let (fields, _) = at.field(RowPlaneRole::Scales, record, Self::WIDTH);
        let mut out = [[[0i32; 8]; 2]; 2];
        #[cfg(target_arch = "x86_64")]
        if V::QUANTIZED_DOTS >= 1 {
            let vectors = unsafe { Self::x86_groups::<GROUP>(fields) }?;
            for (out, vectors) in out.iter_mut().zip(vectors) {
                for (out, vector) in out.iter_mut().zip(vectors) {
                    unsafe { std::arch::x86_64::_mm256_storeu_si256(out.as_mut_ptr().cast(), vector) };
                }
            }
            return Some(out);
        }
        #[cfg(target_arch = "aarch64")]
        if V::QUANTIZED_DOTS >= 1 && Self::X86_GROUPS {
            use std::arch::aarch64::*;
            let lanes = Q8_BLOCK / GROUP;
            let mut bytes = [0u8; 16];
            unsafe { std::ptr::copy_nonoverlapping(fields, bytes.as_mut_ptr(), (lanes * Self::WIDTH).div_ceil(8)) };
            // Lane `i` (field `i`) takes its first byte and the next two,
            // shifted down by its first bit's place: `x86_groups`'s order.
            let (select, shifts) = const {
                let lanes = Q8_BLOCK / GROUP;
                let (mut select, mut shifts) = ([[255u8; 16]; 4], [[0i32; 4]; 4]);
                let mut field = 0;
                while field < lanes {
                    let bit = field * Self::WIDTH;
                    let mut byte = 0;
                    while byte < 3 && bit / 8 + byte < 16 {
                        select[field / 4][4 * (field % 4) + byte] = (bit / 8 + byte) as u8;
                        byte += 1;
                    }
                    shifts[field / 4][field % 4] = -((bit % 8) as i32);
                    field += 1;
                }
                (select, shifts)
            };
            let mut raw = [0u32; 16];
            unsafe {
                let bytes = vld1q_u8(bytes.as_ptr());
                for quarter in 0..lanes / 4 {
                    let words = vreinterpretq_u32_u8(vqtbl1q_u8(bytes, vld1q_u8(select[quarter].as_ptr())));
                    vst1q_u32(raw[4 * quarter..].as_mut_ptr(), vshlq_u32(words, vld1q_s32(shifts[quarter].as_ptr())));
                }
            }
            for (field, raw) in raw.into_iter().enumerate().take(lanes) {
                (out[0][field / 8][field % 8], out[1][field / 8][field % 8]) = Self::split(raw);
            }
            return Some(out);
        }
        let _ = (fields, &mut out);
        None
    }

    #[inline(always)]
    unsafe fn block<const GROUP: usize>(at: At) -> [[(i32, i32); 2]; 8] {
        if at.row.tile.is_some() {
            // A rows8 row's fields lie apart, each read alone.
            let mut fields = [[(0, 0); 2]; 8];
            for (packet, fields) in fields.iter_mut().enumerate() {
                for (half, field) in fields.iter_mut().enumerate() {
                    *field = unsafe { Self::read::<GROUP>(at, at.packet(packet).local::<GROUP>(half)) };
                }
            }
            return fields;
        }
        // The storage group's fields as one word, read once, and each
        // coefficient group's codes from it once.
        let bytes = const {
            let bytes = (Q8_BLOCK / GROUP * Self::WIDTH).div_ceil(8);
            assert!(bytes <= 16, "a storage group's fields fit 16 bytes");
            bytes
        };
        let mut fields = [0u8; 16];
        unsafe {
            std::ptr::copy_nonoverlapping(
                at.plane(RowPlaneRole::Scales),
                fields.as_mut_ptr(),
                bytes,
            )
        };
        let fields = u128::from_le_bytes(fields);
        let mut groups = [(0, 0); 16];
        for (group, codes) in groups.iter_mut().enumerate().take(Q8_BLOCK / GROUP) {
            *codes = Self::split((fields >> (group * Self::WIDTH)) as u32);
        }
        let mut codes = [[(0, 0); 2]; 8];
        for (packet, codes) in codes.iter_mut().enumerate() {
            for (half, codes) in codes.iter_mut().enumerate() {
                *codes = groups[at.packet(packet).local::<GROUP>(half)];
            }
        }
        codes
    }

    #[inline(always)]
    unsafe fn read<const GROUP: usize>(at: At, local: usize) -> (i32, i32) {
        let (bytes, bit) = at.field(RowPlaneRole::Scales, local, Self::WIDTH);
        let mut pair = u32::from(unsafe { *bytes });
        if bit + Self::WIDTH > 8 {
            pair |= u32::from(unsafe { *bytes.add(1) }) << 8;
        }
        Self::codes(pair, bit)
    }
}

/// Local codes carried by grid entries of `ENTRY_BYTES` whole bytes and
/// `HIGH` high-plane bits (`Grid`): bit `j` of a group's code at entry bit
/// `BIT` of the group's entry `j`, a `BITS`-bit scale code read through `L`
/// and then bias `B`'s sign bits.
#[derive(Clone, Copy, Debug)]
pub struct Carried<const BIT: u32, const ENTRY_BYTES: usize, const HIGH: u32, L, const BITS: u32, B>(
    PhantomData<(L, B)>,
);

/// The bias of carried local codes (`registry::LocalBias`): none, or a
/// signed bias.
pub trait CarriedBias: Copy + Send + Sync + 'static {
    const BIAS: Option<LocalBias>;
    /// Sign bits past the scale code: the signed bias's sign.
    const SIGN_BITS: u32 = Self::BIAS.is_some() as u32;
}

/// No bias.
#[derive(Clone, Copy, Debug)]
pub struct Unbiased;

impl PlaneBias for Unbiased {
    const BIAS: Option<LocalBias> = None;
}

impl CarriedBias for Unbiased {
    const BIAS: Option<LocalBias> = None;
}

/// A signed bias (`LocalBias::Signed`): the scale code, negated where the
/// sign bit past it is set, under the factor scaled by `2^EXPONENT`.
#[derive(Clone, Copy, Debug)]
pub struct SignedBias<const EXPONENT: i32>;

impl<const EXPONENT: i32> CarriedBias for SignedBias<EXPONENT> {
    const BIAS: Option<LocalBias> = Some(LocalBias::Signed { exponent: EXPONENT });
}

impl<
        const BIT: u32,
        const ENTRY_BYTES: usize,
        const HIGH: u32,
        L: Interpretation,
        const BITS: u32,
        B: CarriedBias,
    > Locals for Carried<BIT, ENTRY_BYTES, HIGH, L, BITS, B>
{
    const BIAS: Option<LocalBias> = B::BIAS;
    // A `BITS`-bit field's value, negated where a sign bit says so.
    const MAGNITUDE: i32 = 1 << BITS;

    #[inline(always)]
    unsafe fn read<const GROUP: usize>(at: At, local: usize) -> (i32, i32) {
        let width = (BITS + B::SIGN_BITS) as usize;
        // The group's entries from its first, four a packet: bit `j` of the
        // code at entry bit `BIT` of entry `j`, in its whole bytes or past
        // them in its further bits (`grid_entry`).
        let first = (GROUP / 8) * local;
        let code = (0..width).fold(0u32, |code, j| {
            let entry = first + j;
            let at = at.packet(entry / 4);
            let bit = if (BIT as usize) < 8 * ENTRY_BYTES {
                let byte = unsafe { *at.entry(entry % 4, ENTRY_BYTES).add(BIT as usize / 8) };
                u32::from(byte) >> (BIT % 8)
            } else {
                let high = unsafe { grid_high::<HIGH>(at, entry % 4) };
                high >> (BIT as usize - 8 * ENTRY_BYTES)
            };
            code | ((bit & 1) << j)
        });
        let scale = L::value::<BITS>(code & ((1 << BITS) - 1));
        let bias = if B::SIGN_BITS == 0 {
            0
        } else if (code >> BITS) & 1 == 1 {
            -scale
        } else {
            scale
        };
        (scale, bias)
    }
}

/// Hierarchical coefficients (`CodingAxes::Hierarchical`): per storage
/// group (the factor group) a factor of encoding `F` (then with a minimum a
/// minimum factor) in the supers plane, scaled by `2^EXPONENT`; per `GROUP`
/// values local codes `L`. A value's scale is `factor * 2^CODE_EXPONENT *
/// scale code` and its bias the bias factor times the bias code (`L::BIAS`):
/// `± minimum factor * minimum code`, or `factor * 2^(EXPONENT + exponent) *
/// signed code`. A quantized dot sums `scale code * products` (and `bias
/// code * activation sums`) over the factor group exactly, then combines
/// `factor * scaled ± bias factor * biases` once.
#[derive(Clone, Copy, Debug)]
pub struct Hier<F, const EXPONENT: i32, const CODE_EXPONENT: i32, L, const GROUP: usize>(
    PhantomData<(F, L)>,
);

impl<F: Scale, const EXPONENT: i32, const CODE_EXPONENT: i32, L: Locals, const GROUP: usize>
    Hier<F, EXPONENT, CODE_EXPONENT, L, GROUP>
{
    /// Whether the bias factor is a stored minimum factor.
    const MINIMUM: bool = matches!(L::BIAS, Some(LocalBias::Minimum(_)));
    /// Whether the bias term is subtracted.
    const SUBTRACT: bool = matches!(L::BIAS, Some(LocalBias::Minimum(MinimumSign::Subtract)));
    /// `2^(EXPONENT + CODE_EXPONENT)`, folded into the factor.
    const SCALE_POWER: f32 = fold::<F>(EXPONENT + CODE_EXPONENT);
    /// The bias factor's power of two, folded into the minimum factor
    /// (`2^EXPONENT`) or the factor (`2^(EXPONENT + exponent)`).
    const BIAS_POWER: f32 = match L::BIAS {
        Some(LocalBias::Minimum(_)) => fold::<F>(EXPONENT),
        Some(LocalBias::Signed { exponent }) => fold::<F>(EXPONENT + exponent),
        None => 1.0,
    };

    /// The scale factor (`factor * 2^(EXPONENT + CODE_EXPONENT)`) and the
    /// bias factor (the stored minimum factor times `2^EXPONENT`, or the
    /// factor times `2^(EXPONENT + exponent)` of a signed bias) of the
    /// storage group of `at`.
    #[inline(always)]
    unsafe fn factors(at: At) -> (f32, f32) {
        let field = |record| at.field(RowPlaneRole::Supers, record, 8 * F::BYTES).0;
        let minimum = if Self::MINIMUM {
            unsafe { F::read(field(1)) }
        } else {
            0.0
        };
        Self::factors_of(unsafe { F::read(field(0)) }, minimum)
    }

    /// [`Hier::factors`] of the stored factor and minimum factor.
    #[inline(always)]
    fn factors_of(factor: f32, minimum: f32) -> (f32, f32) {
        let bias = match L::BIAS {
            Some(LocalBias::Minimum(_)) => {
                if EXPONENT == 0 { minimum } else { minimum * Self::BIAS_POWER }
            }
            Some(LocalBias::Signed { .. }) => factor * Self::BIAS_POWER,
            None => 0.0,
        };
        let scale = if EXPONENT + CODE_EXPONENT == 0 { factor } else { factor * Self::SCALE_POWER };
        (scale, bias)
    }

    /// The scale codes of packet `at`'s halves (its one group's in both
    /// with one group a packet), read where used.
    ///
    /// # Safety
    /// As [`Locals::read`].
    #[cfg(target_arch = "aarch64")]
    #[inline(always)]
    unsafe fn packet_scales(at: At) -> [i32; 2] {
        let first = unsafe { L::read::<GROUP>(at, at.local::<GROUP>(0)).0 };
        let second = if groups_per_packet(GROUP) == 1 {
            first
        } else {
            unsafe { L::read::<GROUP>(at, at.local::<GROUP>(1)).0 }
        };
        [first, second]
    }

    /// [`Coefficients::dot_q8`] of `N` rows over the block's first `packets`
    /// packets, with the activation blocks' code sums `sums`: one row, or
    /// rows `g` and `g + 8` of an `mma16` tile (`Row::partners`), whose
    /// shared code bytes each k-block reads once for both
    /// (`Codes::neon_partners`).
    #[inline(always)]
    unsafe fn dot_rows<C: Codes, const N: usize, const M: usize, V: Vectors>(
        rows: [&Row; N],
        b: usize,
        packets: usize,
        x: [&Q8Block; M],
        sums: &[[[i32; 2]; 8]; M],
    ) -> [[f32; M]; N] {
        const { assert!(N == 1 || N == 2, "a row or a row pair") };
        let halves = const { groups_per_packet(GROUP) };
        let whole = halves == 1;
        let mut groups = [At::group(rows[0], b); N];
        for (group, row) in groups.iter_mut().zip(rows) {
            *group = At::group(row, b);
        }
        // The block's codes, read once: the minimum terms', and the scale
        // codes that x86 broadcasts from memory (a load operand). NEON reads
        // each packet's scale codes where used instead: its broadcast from a
        // general register replaces the block's round trip through memory.
        let mut codes = [unsafe { L::block::<GROUP>(groups[0]) }; N];
        for n in 1..N {
            codes[n] = unsafe { L::block::<GROUP>(groups[n]) };
        }
        let mut scaled = [[0i32; M]; N];
        let mut minimums = [[0i32; M]; N];
        // x86 takes the codes less `least` (`packed_x86::shift`), and adds
        // back `least Σ scale Σ a` per group with the minimum terms.
        #[cfg(target_arch = "x86_64")]
        let least = if V::QUANTIZED_DOTS >= 1 {
            const {
                match packed_x86::shift::<C>(packed_x86::Products::Bytes16) {
                    Some(least) => least,
                    None => 0,
                }
            }
        } else {
            0
        };
        #[cfg(not(target_arch = "x86_64"))]
        let least = 0;
        if Self::BIASED || least != 0 {
            for n in 0..N {
                for m in 0..M {
                    for local in packets_of(packets) {
                        for h in 0..halves {
                            let ((scale, minimum), sum) = (codes[n][local][h], sums[m][local][h]);
                            if Self::BIASED {
                                minimums[n][m] += minimum * sum;
                            }
                            if least != 0 {
                                scaled[n][m] += least * scale * sum;
                            }
                        }
                    }
                }
            }
        }
        #[cfg(target_arch = "aarch64")]
        if V::QUANTIZED_DOTS >= 1 {
            unsafe {
                use std::arch::aarch64::*;
                // Each half accumulates in its own lanes: one multiply-add per
                // half, which the compiler cannot reassociate into a multiply,
                // a multiply-add and an add.
                let mut lanes = [[[vdupq_n_s32(0); 2]; M]; N];
                // A k-block's two packets read together (`Codes::neon_pair`;
                // the group's packets are whole k-blocks), of both rows of a
                // pair at once.
                for pair in packets_of(packets.div_ceil(2)) {
                    let first = 2 * pair;
                    let weights: [[[int8x16_t; 2]; 2]; N] = if N == 2 {
                        let both = C::neon_partners(groups[0].packet(first), groups[N - 1].packet(first));
                        std::array::from_fn(|n| both[n])
                    } else {
                        [C::neon_pair(groups[0].packet(first)); N]
                    };
                    for n in 0..N {
                        for (local, weights) in (first..(first + 2).min(packets)).zip(weights[n]) {
                            let [first_code, second_code] = Self::packet_scales(groups[n].packet(local));
                            let zero = vdupq_n_s32(0);
                            for (lanes, x) in lanes[n].iter_mut().zip(x.iter()) {
                                let activations = x.codes[PACKET * local..].as_ptr();
                                let (first, second) = (vld1q_s8(activations), vld1q_s8(activations.add(16)));
                                if whole {
                                    let products = packed_neon::sdot(
                                        packed_neon::sdot(zero, weights[0], first),
                                        weights[1],
                                        second,
                                    );
                                    lanes[0] = vmlaq_n_s32(lanes[0], products, first_code);
                                } else {
                                    lanes[0] = vmlaq_n_s32(
                                        lanes[0],
                                        packed_neon::sdot(zero, weights[0], first),
                                        first_code,
                                    );
                                    lanes[1] = vmlaq_n_s32(
                                        lanes[1],
                                        packed_neon::sdot(zero, weights[1], second),
                                        second_code,
                                    );
                                }
                            }
                        }
                    }
                }
                for (scaled, lanes) in scaled.iter_mut().zip(lanes) {
                    for (scaled, [first, second]) in scaled.iter_mut().zip(lanes) {
                        *scaled += vaddvq_s32(vaddq_s32(first, second));
                    }
                }
            }
        }
        #[cfg(target_arch = "x86_64")]
        if V::QUANTIZED_DOTS >= 1 {
            unsafe {
                use std::arch::x86_64::*;
                // The pair products take one 16-bit scale code per half.
                let shift = const { packed_x86::shift::<C>(packed_x86::Products::Bytes16) };
                let mut lanes = [[_mm256_setzero_si256(); M]; N];
                // A k-block's two packets read together (`Codes::x86_pair`;
                // the group's packets are whole k-blocks), of both rows of a
                // pair at once.
                for pair in packets_of(packets.div_ceil(2)) {
                    let first = 2 * pair;
                    let weights: [[__m256i; 2]; N] = if N == 2 {
                        let both = C::x86_partners(groups[0].packet(first), groups[N - 1].packet(first));
                        std::array::from_fn(|n| both[n])
                    } else {
                        [C::x86_pair(groups[0].packet(first)); N]
                    };
                    for n in 0..N {
                        for (local, weights) in (first..(first + 2).min(packets)).zip(weights[n]) {
                            let (first, second) = (codes[n][local][0].0, codes[n][local][1].0);
                            let scales = if whole {
                                _mm256_set1_epi16(first as i16)
                            } else {
                                _mm256_set_m128i(
                                    _mm_set1_epi16(second as i16),
                                    _mm_set1_epi16(first as i16),
                                )
                            };
                            for m in 0..M {
                                let activations =
                                    packed_x86::load(x[m].codes.as_ptr().add(PACKET * local));
                                lanes[n][m] = match shift {
                                    Some(least) => packed_x86::dot_scaled(lanes[n][m], packed_x86::operands(weights, least), activations, scales),
                                    None => packed_x86::dot_scaled(
                                        lanes[n][m],
                                        _mm256_abs_epi8(weights),
                                        _mm256_sign_epi8(activations, weights),
                                        scales,
                                    ),
                                };
                            }
                        }
                    }
                }
                for (scaled, lanes) in scaled.iter_mut().zip(lanes) {
                    for (scaled, lanes) in scaled.iter_mut().zip(lanes) {
                        *scaled += packed_x86::sum(lanes);
                    }
                }
            }
        }
        if V::QUANTIZED_DOTS == 0 {
            for n in 0..N {
                for local in packets_of(packets) {
                    let values = unsafe { C::values(groups[n].packet(local)) };
                    let exact = scalar_dots::<M, GROUP, V>(&values, x, local);
                    for m in 0..M {
                        for h in 0..halves {
                            scaled[n][m] += codes[n][local][h].0 * exact[m][h];
                        }
                    }
                }
            }
        }
        let mut out = [[0.0f32; M]; N];
        for n in 0..N {
            let factors = unsafe { Self::factors(groups[n]) };
            for m in 0..M {
                out[n][m] = x[m].d * Self::combine(factors, scaled[n][m], minimums[n][m]);
            }
        }
        out
    }

    /// [`Coefficients::dot_q8`] of the 16 rows of one `mma16` tile
    /// (row 0 `row`, rows `stride` apart) of nibble-lane codes (`Codes::NIBBLE_LANES`)
    /// against block `b` of the activations on x86, over whole k-blocks. A
    /// k-block's 64 code bytes of row pair `g` hold, in byte `16t + 4s +
    /// j`, row `g`'s code (low nibble) and row `g + 8`'s (high nibble) of
    /// column `16s + [2t, 2t + 8, 2t + 1, 2t + 9][j]` (`code_bit`); so the
    /// activations are permuted into that order (`activations`, once a call,
    /// `permuted_activations`), and every pair's bytes are multiplied where
    /// they lie, both rows at once, into each row's exact sum of every step
    /// of 16 columns, which [`Hier::combine_tile16`] turns into
    /// [`Hier::dot_rows`]'s result bit for bit.
    #[cfg(target_arch = "x86_64")]
    #[inline(always)]
    unsafe fn dot_tile16<C: Codes>(
        row: &Row,
        stride: usize,
        b: usize,
        packets: usize,
        x: &Q8Block,
        activations: &[[std::arch::x86_64::__m256i; 2]],
        groups: &[std::arch::x86_64::__m256i; 2],
    ) -> [f32; 16] {
        use std::arch::x86_64::*;
        let (high, offset) = C::NIBBLE_LANES.expect("nibble lanes");
        let blocks = packets / 2;
        let mma16 = row.mma16.as_ref().expect("an mma16 tile");
        let first = (mma16.first_packet + b * mma16.group_packets) / 2;
        let [low_plane, high_plane] = &mma16.planes;
        // Each k-block's lanes `4g ..4g + 4` of each code plane: together
        // (`Row::lanes_whole`), wherever the k-block's lanes lie.
        let mut low = [[std::ptr::null(); 8]; 4];
        let mut highs = [[std::ptr::null(); 8]; 4];
        for block in 0..blocks {
            let lanes = 32 * (first + block);
            for (plane, quads) in [(low_plane, &mut low[block]), (high_plane, &mut highs[block])].into_iter().take(1 + (high > 0) as usize) {
                *quads = plane.quads(lanes);
                // Lines read ahead, so a cold matrix streams instead of
                // stalling a call at a time: this tile's next storage
                // group's same lanes, which this call reads next, and the
                // next tile's, which the next tile reads (a prefetch past
                // the matrix is a hint, never a fault).
                let next = plane.quads(lanes + 128);
                for g in 0..8 {
                    for at in [next[g], quads[g].wrapping_add(16 * plane.stride)] {
                        unsafe { _mm_prefetch::<_MM_HINT_T0>(at.cast()) };
                    }
                }
            }
        }
        // Each row's step sums, k-block `block`'s four in lanes `4 block`
        // on: `[steps 0..8, steps 8..16]`. Two row pairs an iteration, so
        // their loads and multiplies overlap.
        let mut steps = [[unsafe { _mm_setzero_si128() }; 4]; 16];
        for block in 0..blocks {
            let activations = activations[block];
            for pairs in 0..4 {
                unsafe {
                    let nibble = _mm256_set1_epi8(0x0f);
                    let mut products = [[_mm256_setzero_si256(); 2]; 2];
                    for half in 0..2 {
                        for (pair, products) in products.iter_mut().enumerate() {
                            let g = 2 * pairs + pair;
                            // Lanes `4g + 2 half` and the next.
                            let bytes = _mm256_loadu_si256(low[block][g].add(2 * half * low_plane.unit).cast());
                            let mut first = _mm256_and_si256(bytes, nibble);
                            let mut second = _mm256_and_si256(_mm256_srli_epi16::<4>(bytes), nibble);
                            if high > 0 {
                                let [first_high, second_high] = high_bits(highs[block][g].add(2 * half * high_plane.unit), high);
                                first = _mm256_or_si256(first, first_high);
                                second = _mm256_or_si256(second, second_high);
                            }
                            // Codes below 64 times activations of at most
                            // 127: two products a pair and two pairs fit 16
                            // bits.
                            products[0] = _mm256_add_epi16(products[0], _mm256_maddubs_epi16(first, activations[half]));
                            products[1] = _mm256_add_epi16(products[1], _mm256_maddubs_epi16(second, activations[half]));
                        }
                    }
                    let ones = _mm256_set1_epi16(1);
                    for (pair, products) in products.iter().enumerate() {
                        for upper in 0..2 {
                            // Lanes `[s0..s4 | s0..s4]` of lanes t and
                            // t + 1: their halves added, the k-block's four
                            // steps.
                            let lanes = _mm256_madd_epi16(products[upper], ones);
                            steps[2 * pairs + pair + 8 * upper][block] =
                                _mm_add_epi32(_mm256_castsi256_si128(lanes), _mm256_extracti128_si256::<1>(lanes));
                        }
                    }
                }
            }
        }
        unsafe { Self::combine_tile16(row, stride, b, x, &steps, groups, offset) }
    }

    /// The 16 rows' block `b` sums of a tile from each row's exact code
    /// step sums `steps` (row `r`'s k-block `block` four steps of 16
    /// columns, in column order, at `steps[r][block]`): row `r` at `row`
    /// plus `r` strides, its raw codes the values less `offset`, `groups`
    /// the block's activation code sum of each coefficient group. Each
    /// row's coefficient codes are decoded as vectors (`Locals::x86_groups`)
    /// and applied to its sixteen step sums, its minimum and offset terms to
    /// the group sums, all in i32 (the exact integers [`Hier::dot_rows`]
    /// sums), then eight rows' factors read and combined at once, each
    /// lane's arithmetic `Self::combine`'s: [`Hier::dot_rows`]'s result bit
    /// for bit.
    #[cfg(target_arch = "x86_64")]
    #[inline(always)]
    unsafe fn combine_tile16(
        row: &Row,
        stride: usize,
        b: usize,
        x: &Q8Block,
        steps: &[[std::arch::x86_64::__m128i; 4]; 16],
        groups: &[std::arch::x86_64::__m256i; 2],
        offset: i32,
    ) -> [f32; 16] {
        use std::arch::x86_64::*;
        // Each row's coefficient terms in i32, then eight rows' factors and
        // combine at once, each lane as `Self::combine` computes it.
        let group = At::group(row, b);
        let (scales_plane, supers) = (group.plane(RowPlaneRole::Scales), group.plane(RowPlaneRole::Supers));
        let mut scaled = [0i32; 16];
        let mut minimums = [0i32; 16];
        for r in 0..16 {
            unsafe {
                let [scales, minimum_codes] = L::x86_groups::<GROUP>(scales_plane.add(r * stride)).expect("vector coefficient codes");
                // Each step's scale code: a group of 32 is two steps.
                let step_scales = if GROUP == 16 {
                    scales
                } else {
                    [
                        _mm256_permutevar8x32_epi32(scales[0], _mm256_setr_epi32(0, 0, 1, 1, 2, 2, 3, 3)),
                        _mm256_permutevar8x32_epi32(scales[0], _mm256_setr_epi32(4, 4, 5, 5, 6, 6, 7, 7)),
                    ]
                };
                let mut sum = _mm256_setzero_si256();
                for (pair, step_scales) in step_scales.iter().enumerate() {
                    let sums = _mm256_set_m128i(steps[r][2 * pair + 1], steps[r][2 * pair]);
                    sum = _mm256_add_epi32(sum, _mm256_mullo_epi32(sums, *step_scales));
                }
                let mut terms = [_mm256_setzero_si256(); 2];
                for half in 0..32 / GROUP {
                    if offset != 0 {
                        terms[0] = _mm256_add_epi32(terms[0], _mm256_mullo_epi32(scales[half], groups[half]));
                    }
                    if Self::BIASED {
                        terms[1] = _mm256_add_epi32(terms[1], _mm256_mullo_epi32(minimum_codes[half], groups[half]));
                    }
                }
                scaled[r] = sum_lanes(sum) + offset * sum_lanes(terms[0]);
                minimums[r] = sum_lanes(terms[1]);
            }
        }
        let mut out = [0.0f32; 16];
        for eight in 0..2 {
            unsafe {
                let supers = supers.add(8 * eight * stride);
                let factor = F::read8(supers, stride);
                let scale = if CODE_EXPONENT == 0 && EXPONENT == 0 {
                    factor
                } else {
                    _mm256_mul_ps(factor, _mm256_set1_ps(Self::SCALE_POWER))
                };
                // Lanes 8 * eight.. of the i32 sums, as f32 (no closure: it
                // would not take the caller's features).
                let mut value = _mm256_mul_ps(scale, _mm256_cvtepi32_ps(_mm256_loadu_si256(scaled.as_ptr().add(8 * eight).cast())));
                if Self::BIASED {
                    let bias = match L::BIAS {
                        Some(LocalBias::Minimum(_)) => {
                            let minimum = F::read8(supers.add(F::BYTES), stride);
                            if EXPONENT == 0 { minimum } else { _mm256_mul_ps(minimum, _mm256_set1_ps(Self::BIAS_POWER)) }
                        }
                        Some(LocalBias::Signed { .. }) => _mm256_mul_ps(factor, _mm256_set1_ps(Self::BIAS_POWER)),
                        None => _mm256_setzero_ps(),
                    };
                    let term = _mm256_mul_ps(bias, _mm256_cvtepi32_ps(_mm256_loadu_si256(minimums.as_ptr().add(8 * eight).cast())));
                    value = if Self::SUBTRACT { _mm256_sub_ps(value, term) } else { _mm256_add_ps(value, term) };
                }
                _mm256_storeu_ps(out.as_mut_ptr().add(8 * eight), _mm256_mul_ps(_mm256_set1_ps(x.d), value));
            }
        }
        out
    }

    /// [`Coefficients::dot_q8_tiles`] on x86: every 16 rows of `out` a tile
    /// (`Self::dot_tile16`), the activations permuted and the group sums
    /// formed once for every tile, each tile's blocks summed in block order
    /// as [`dot_q8_rows`].
    #[cfg(target_arch = "x86_64")]
    #[inline(always)]
    unsafe fn tiles16<C: Codes>(row: &Row, stride: usize, x: &[Q8Block], k: usize, out: &mut [f32]) {
        use std::arch::x86_64::*;
        // Blocks a chunk at a time, on the stack: each chunk's activations
        // permuted and group sums formed once for every tile.
        const CHUNK: usize = 8;
        out.fill(0.0);
        for (chunk, blocks) in x.chunks(CHUNK).enumerate() {
            let mut activations = [[unsafe { _mm256_setzero_si256() }; 2]; 4 * CHUNK];
            let mut groups = [[unsafe { _mm256_setzero_si256() }; 2]; CHUNK];
            for (index, block) in blocks.iter().enumerate() {
                for kblock in 0..4 {
                    activations[4 * index + kblock] = unsafe { permuted_activations(block.codes.as_ptr().add(64 * kblock)) };
                }
                groups[index] = unsafe { Self::group_sums(block) };
            }
            for (tile, out) in out.chunks_exact_mut(16).enumerate() {
                let first = row.shifted(16 * tile * stride);
                for (index, block) in blocks.iter().enumerate() {
                    let b = CHUNK * chunk + index;
                    let packets = block_packets(b, k).len();
                    let values = unsafe { Self::dot_tile16::<C>(&first, stride, b, packets, block, &activations[4 * index..], &groups[index]) };
                    for (sum, value) in out.iter_mut().zip(values) {
                        *sum += value;
                    }
                }
            }
        }
    }

    /// [`Coefficients::dot_q8_tiles`] on x86 of rows each whole in its own
    /// bytes (`Placement::Rows`: `rows16`), from `first`, whose values less
    /// `least` are unsigned bytes (`packed_x86::shift`): every 16 rows of
    /// `out` a tile, each row's exact step sums of every k-block formed from
    /// its shifted values in column order (`Codes::x86_pair`) against the
    /// activations where they lie, then [`Hier::combine_tile16`]; each
    /// tile's blocks summed in block order as [`dot_q8_rows`].
    #[cfg(target_arch = "x86_64")]
    #[inline(always)]
    unsafe fn tiles_rows<C: Codes>(first: *const u8, geometry: &RowGeometry, x: &[Q8Block], k: usize, out: &mut [f32], least: i32) {
        use std::arch::x86_64::*;
        out.fill(0.0);
        for (tile, out) in out.chunks_exact_mut(16).enumerate() {
            let rows = geometry.rows::<16>(first.wrapping_add(16 * tile * geometry.stride));
            for (b, block) in x.iter().enumerate() {
                let kblocks = block_packets(b, k).len() / 2;
                // No closures here: a closure would not take the caller's
                // features, and its intrinsics would compile as calls.
                let mut activations = [unsafe { _mm256_setzero_si256() }; Q8_BLOCK / PACKET];
                for (p, activation) in activations.iter_mut().enumerate() {
                    *activation = unsafe { packed_x86::load(block.codes.as_ptr().add(PACKET * p)) };
                }
                let mut steps = [[unsafe { _mm_setzero_si128() }; 4]; 16];
                for (steps, row) in steps.iter_mut().zip(&rows) {
                    let group = At::group(row, b);
                    for (kblock, steps) in steps.iter_mut().enumerate().take(kblocks) {
                        unsafe {
                            let ones = _mm256_set1_epi16(1);
                            // Per 32-bit lane, four columns' products: lanes
                            // 0..4 a packet's step 0, lanes 4..8 its step 1.
                            let [first, second] = C::x86_pair(group.packet(2 * kblock));
                            let [first, second] = [packed_x86::operands(first, least), packed_x86::operands(second, least)];
                            let [first, second] = [
                                _mm256_madd_epi16(_mm256_maddubs_epi16(first, activations[2 * kblock]), ones),
                                _mm256_madd_epi16(_mm256_maddubs_epi16(second, activations[2 * kblock + 1]), ones),
                            ];
                            // Each 128-bit half summed within: low
                            // `[s0, s2, ..]`, high `[s1, s3, ..]`.
                            let pairs = _mm256_hadd_epi32(first, second);
                            let sums = _mm256_hadd_epi32(pairs, pairs);
                            *steps = _mm_unpacklo_epi32(_mm256_castsi256_si128(sums), _mm256_extracti128_si256::<1>(sums));
                        }
                    }
                }
                let values =
                    unsafe { Self::combine_tile16(&rows[0], geometry.stride, b, block, &steps, &Self::group_sums(block), least) };
                for (sum, value) in out.iter_mut().zip(values) {
                    *sum += value;
                }
            }
        }
    }

    /// The activation code sum of each coefficient group of `block`, as
    /// [`Hier::combine_tile16`] takes them.
    #[cfg(target_arch = "x86_64")]
    #[inline(always)]
    unsafe fn group_sums(block: &Q8Block) -> [std::arch::x86_64::__m256i; 2] {
        use std::arch::x86_64::*;
        unsafe {
            let sums = _mm256_loadu_si256(block.sums.as_ptr().cast());
            if GROUP == 16 {
                [_mm256_cvtepi16_epi32(_mm256_castsi256_si128(sums)), _mm256_cvtepi16_epi32(_mm256_extracti128_si256::<1>(sums))]
            } else {
                [_mm256_madd_epi16(sums, _mm256_set1_epi16(1)), _mm256_setzero_si256()]
            }
        }
    }

    /// [`Coefficients::dot_q8`] over the block's first `packets` packets.
    #[inline(always)]
    unsafe fn dot_block<C: Codes, const R: usize, const M: usize, V: Vectors>(
        rows: &[Row; R],
        b: usize,
        packets: usize,
        x: [&Q8Block; M],
    ) -> [[f32; M]; R] {
        let whole = const { groups_per_packet(GROUP) } == 1;
        // Each activation block's code sums, per packet and half (the whole
        // packet in both halves with one group).
        let mut sums = [[[0i32; 2]; 8]; M];
        for (sums, x) in sums.iter_mut().zip(x.iter()) {
            for (local, sums) in sums.iter_mut().enumerate().take(packets) {
                for (h, sum) in sums.iter_mut().enumerate() {
                    *sum = activation_sum(x, local, h, whole);
                }
            }
        }
        let mut out = [[0.0f32; M]; R];
        // An mma16 tile's row pairs, each pair's shared bytes read once.
        if Row::partners(rows) {
            for g in 0..R / 2 {
                let [first, second] =
                    unsafe { Self::dot_rows::<C, 2, M, V>([&rows[g], &rows[g + R / 2]], b, packets, x, &sums) };
                (out[g], out[g + R / 2]) = (first, second);
            }
            return out;
        }
        // Row by row, each whole.
        for (out, row) in out.iter_mut().zip(rows) {
            [*out] = unsafe { Self::dot_rows::<C, 1, M, V>([row], b, packets, x, &sums) };
        }
        out
    }

    /// The (scale, bias) of a group of codes `(scale, minimum)` under
    /// `factors`.
    #[inline(always)]
    fn coefficient(factors: (f32, f32), (scale, minimum): (i32, i32)) -> (f32, f32) {
        let bias = if Self::BIASED {
            let bias = factors.1 * minimum as f32;
            if Self::SUBTRACT {
                -bias
            } else {
                bias
            }
        } else {
            0.0
        };
        (factors.0 * scale as f32, bias)
    }

    /// `factor * scaled ± bias factor * biases`, the bias term's product
    /// rounded on its own.
    #[inline(always)]
    fn combine(factors: (f32, f32), scaled: i32, minimums: i32) -> f32 {
        let scaled = factors.0 * scaled as f32;
        if !Self::BIASED {
            scaled
        } else if Self::SUBTRACT {
            scaled - factors.1 * minimums as f32
        } else {
            scaled + factors.1 * minimums as f32
        }
    }
}

impl<F: Scale, const EXPONENT: i32, const CODE_EXPONENT: i32, L: Locals, const GROUP: usize>
    Coefficients for Hier<F, EXPONENT, CODE_EXPONENT, L, GROUP>
{
    const BIASED: bool = L::BIAS.is_some();
    const UNIT: usize = if GROUP < PACKET { GROUP } else { PACKET };
    const INTEGER_UNITS: bool = true;
    const MAGNITUDE: i32 = L::MAGNITUDE;
    type Units<const W: usize> = tile::Scaled<W>;

    #[inline(always)]
    unsafe fn tile_side<const STORAGE: usize, V: Vectors>(rows: &[Row; 8], b: usize, packets: usize, panel: bool, side: &mut tile::Side) {
        let units = packets * PACKET / Self::UNIT;
        // Each record's eight rows together in its span (`At::field`): a
        // record a unit, one or two decoded at a time. (No closures: a
        // closure would not take the caller's features.)
        let group = At::group(&rows[0], b);
        let mut vectors = panel && (GROUP == 16 || GROUP == 32);
        for first in (0..if vectors { units } else { 0 }).step_by((Q8_BLOCK / GROUP / 8).max(1)) {
            let Some([scales, minimums]) = (unsafe { L::groups8::<GROUP, V>(group, first) }) else {
                vectors = false;
                break;
            };
            for (v, (scales, minimums)) in scales.into_iter().zip(minimums).enumerate().take(Q8_BLOCK / GROUP / 8) {
                (side.widen[first + v], side.minimums[first + v]) = (scales, minimums);
            }
        }
        if vectors {
            let (factor, minimum) = (
                group.field(RowPlaneRole::Supers, 0, 8 * F::BYTES).0,
                group.field(RowPlaneRole::Supers, 1, 8 * F::BYTES).0,
            );
            let factors = unsafe { tile::read8::<F, V>(factor) };
            let minimums = if Self::MINIMUM { unsafe { tile::read8::<F, V>(minimum) } } else { [0.0; 8] };
            for r in 0..8 {
                (side.factors[0][r], side.factors[1][r]) = Self::factors_of(factors[r], minimums[r]);
            }
            return;
        }
        for (r, row) in rows.iter().enumerate() {
            let group = At::group(row, b);
            let codes = unsafe { L::block::<GROUP>(group) };
            for u in 0..units {
                let (packet, half) = (u * Self::UNIT / PACKET, u * Self::UNIT % PACKET / 16);
                (side.widen[u][r], side.minimums[u][r]) = codes[packet][half];
            }
            (side.factors[0][r], side.factors[1][r]) = unsafe { Self::factors(group) };
        }
    }

    #[inline(always)]
    fn tile_combine<const W: usize, const R: usize>(side: &tile::Side, units: &tile::Scaled<W>, x: &Q8Tile<R>) -> [[f32; 8]; W] {
        let mut out = [[0.0f32; 8]; W];
        for (a, out) in out.iter_mut().enumerate() {
            // Each row's exact `scaled` and `minimums`, as `Hier::dot_rows`.
            let (scaled, minimums) = (&units.scaled[a], &units.minimums[a]);
            for r in 0..8 {
                out[r] = x.d[a] * Self::combine((side.factors[0][r], side.factors[1][r]), scaled[r], minimums[r]);
            }
        }
        out
    }

    /// On x86 with quantized dots, rows whose coefficient codes decode as
    /// vectors (`Locals::X86_GROUPS`), over whole k-blocks: `mma16` tiles of
    /// nibble-lane codes (`Codes::NIBBLE_LANES`), from a tile's first row,
    /// in matrices of whole tiles ([`Hier::tiles16`]); `rows16` rows of
    /// unsigned byte values, from any row ([`Hier::tiles_rows`]).
    #[inline(always)]
    unsafe fn dot_q8_tiles<C: Codes, V: Vectors>(first: *const u8, geometry: &RowGeometry, x: &[Q8Block], k: usize, out: &mut [f32]) -> bool {
        #[cfg(target_arch = "x86_64")]
        if V::QUANTIZED_DOTS >= 1 && L::X86_GROUPS && (GROUP == 16 || GROUP == 32) && k % 64 == 0 && out.len() % 16 == 0 {
            match geometry.placement {
                Placement::Mma16 if C::NIBBLE_LANES.is_some() && geometry.matrix_rows % 16 == 0 => {
                    let row = geometry.row(first);
                    if row.lanes_whole() && row.mma16.as_ref().is_some_and(|mma16| mma16.row == 0) {
                        unsafe { Self::tiles16::<C>(&row, geometry.stride, x, k, out) };
                        return true;
                    }
                }
                Placement::Rows => {
                    if let Some(least) = const { packed_x86::shift::<C>(packed_x86::Products::Bytes16) } {
                        unsafe { Self::tiles_rows::<C>(first, geometry, x, k, out, least) };
                        return true;
                    }
                }
                _ => {}
            }
        }
        let _ = (first, geometry, x, k, out);
        false
    }

    #[cfg(test)]
    unsafe fn reference<C: Codes, const STORAGE: usize>(row: &Row, b: usize, packets: usize, x: &Q8Block) -> f32 {
        // `scaled = Σ_g s_g Σ v a` and `minimums = Σ_g m_g Σ a` exactly,
        // then `d * combine`.
        let group = At::group(row, b);
        let (mut scaled, mut minimums) = (0i32, 0i32);
        for local in packets_of(packets) {
            let at = group.packet(local);
            let values = unsafe { C::values(at) };
            for (i, value) in values.into_iter().enumerate() {
                let (scale, minimum) = unsafe { L::read::<GROUP>(at, at.local::<GROUP>(i / 16)) };
                let code = i32::from(x.codes[PACKET * local + i]);
                scaled += scale * i32::from(value) * code;
                minimums += minimum * code;
            }
        }
        x.d * Self::combine(unsafe { Self::factors(group) }, scaled, minimums)
    }

    #[inline(always)]
    unsafe fn decode<const STORAGE: usize>(at: At) -> [(f32, f32); 2] {
        let factors = unsafe { Self::factors(at) };
        let first = unsafe { L::read::<GROUP>(at, at.local::<GROUP>(0)) };
        let first = Self::coefficient(factors, first);
        if GROUP == 16 {
            let second = unsafe { L::read::<GROUP>(at, at.local::<GROUP>(1)) };
            [first, Self::coefficient(factors, second)]
        } else {
            [first, first]
        }
    }

    #[inline(always)]
    unsafe fn dot_q8<C: Codes, const STORAGE: usize, const R: usize, const M: usize, V: Vectors>(
        rows: &[Row; R],
        b: usize,
        packets: usize,
        x: [&Q8Block; M],
    ) -> [[f32; M]; R] {
        const {
            assert!(
                STORAGE == Q8_BLOCK,
                "a factor group is one activation block"
            )
        };
        // A factor group is the whole block. Every block but a row's last
        // has all its packets: its own instance bounds the packet loops by
        // a constant, which unrolls the block into registers.
        let full = Q8_BLOCK / PACKET;
        if packets == full {
            unsafe { Self::dot_block::<C, R, M, V>(rows, b, full, x) }
        } else {
            unsafe { Self::dot_block::<C, R, M, V>(rows, b, packets, x) }
        }
    }
}

/// Packed rows of `STORAGE`-value storage groups whose packets' values are
/// codes `C` scaled by coefficients `K`.
#[derive(Clone, Copy, Debug)]
pub struct PackedRows<C, K, const STORAGE: usize>(PhantomData<(C, K)>);

impl<C: Codes, K: Coefficients, const STORAGE: usize> Format for PackedRows<C, K, STORAGE> {
    #[inline(always)]
    unsafe fn decode<V: Vectors>(row: &Row, p: usize, _k: usize, out: &mut [f32; PACKET]) {
        const { assert!(STORAGE % PACKET == 0, "storage groups of whole packets") };
        let at = At::of::<STORAGE>(row, p);
        let coefficients = unsafe { K::decode::<STORAGE>(at) };
        #[cfg(target_arch = "aarch64")]
        unsafe {
            let values = C::neon(at);
            for half in 0..2 {
                let (scale, bias) = coefficients[half];
                let out = out.as_mut_ptr().add(16 * half);
                if K::BIASED {
                    packed_neon::affine_store(values[half], scale, bias, out);
                } else {
                    packed_neon::scaled_store(values[half], scale, out);
                }
            }
        }
        #[cfg(not(target_arch = "aarch64"))]
        {
            let values = unsafe { C::values(at) };
            for half in 0..2 {
                let (scale, bias) = coefficients[half];
                for i in 16 * half..16 * half + 16 {
                    out[i] = if K::BIASED {
                        scale.mul_add(f32::from(values[i]), bias)
                    } else {
                        scale * f32::from(values[i])
                    };
                }
            }
        }
    }

    #[inline(always)]
    unsafe fn dot_q8<const R: usize, const M: usize, V: Vectors>(
        rows: &[Row; R],
        b: usize,
        k: usize,
        x: [&Q8Block; M],
    ) -> [[f32; M]; R] {
        let packets = block_packets(b, k).len();
        unsafe { K::dot_q8::<C, STORAGE, R, M, V>(rows, b, packets, x) }
    }

    #[inline(always)]
    unsafe fn dot_q8_tiles<V: Vectors>(first: *const u8, geometry: &RowGeometry, x: &[Q8Block], k: usize, out: &mut [f32]) -> bool {
        unsafe { K::dot_q8_tiles::<C, V>(first, geometry, x, k, out) }
    }

    const TILED: bool = true;

    #[cfg(test)]
    unsafe fn reference_q8(row: &Row, k: usize, x: &[Q8Block]) -> f32 {
        (0..crate::quant::blocks(k)).fold(0.0, |sum, b| {
            sum + unsafe { K::reference::<C, STORAGE>(row, b, block_packets(b, k).len(), &x[b]) }
        })
    }

    #[inline(always)]
    unsafe fn tile_q8<V: Vectors, const R: usize>(first: *const u8, geometry: &RowGeometry, n: usize, x: &[Q8Tile<R>], m: usize, k: usize, out: &mut [f32]) {
        unsafe { tile::tile::<C, K, STORAGE, V, R>(first, geometry, n, x, m, k, out) }
    }
}

/// A packet's 32 floating codes as exact `f32`s: a floating code component
/// (`registry::ValueCodes::E4m3`), whose values no signed byte holds.
pub trait FloatCodes: Copy + Send + Sync + 'static {
    /// The values of packet `at`, in element order, a NaN code NaN.
    ///
    /// # Safety
    /// `at` is a packet of a row.
    unsafe fn values(at: At) -> [f32; PACKET];
}

/// E4M3 bytes: each code its exact value, its sign kept on a zero, both NaN
/// codes (`S.1111.111`) the registry's one NaN (`f32::NAN`, the canonical
/// NaN its arithmetic gives), which every product with a scale propagates.
/// They are the registry's decoded values (its decode axes' code exponent
/// 0), so a value is one product of the code's value and its scale, as the
/// registry decodes it, under every scale.
#[derive(Clone, Copy, Debug)]
pub struct E4m3;

impl E4m3 {
    const VALUES: [f32; 256] = {
        let mut values = [0.0f32; 256];
        let mut raw = 0;
        while raw < 256 {
            let (exponent, mantissa) = ((raw >> 3) & 15, raw & 7);
            // (8 + m) 2^(e - 10) for a normal code, m 2^-9 for a subnormal
            // one.
            let magnitude = match (exponent, mantissa) {
                (15, 7) => f32::NAN,
                (0, _) => mantissa as f32 * power2(-9),
                _ => ((8 + mantissa) << (exponent - 1)) as f32 * power2(-9),
            };
            values[raw] = if raw >= 128 && !magnitude.is_nan() { -magnitude } else { magnitude };
            raw += 1;
        }
        values
    };
}

impl FloatCodes for E4m3 {
    #[inline(always)]
    unsafe fn values(at: At) -> [f32; PACKET] {
        let codes: [u8; PACKET] = match at.code(RowPlaneRole::Codes) {
            CodeBytes::Packet(bytes) => unsafe { read(bytes) },
            CodeBytes::Lanes(lanes) => unsafe { lane_codes(lanes, 8) },
            CodeBytes::Panel(units) => unsafe { panel_codes(units, 8) },
        };
        codes.map(|code| Self::VALUES[usize::from(code)])
    }
}

/// Packed rows of `STORAGE`-value storage groups whose packets' values are
/// floating codes `C` scaled by coefficients `K`.
#[derive(Clone, Copy, Debug)]
pub struct FloatRows<C, K, const STORAGE: usize>(PhantomData<(C, K)>);

impl<C: FloatCodes, K: Coefficients, const STORAGE: usize> Format for FloatRows<C, K, STORAGE> {
    #[inline(always)]
    unsafe fn decode<V: Vectors>(row: &Row, p: usize, _k: usize, out: &mut [f32; PACKET]) {
        const { assert!(STORAGE % PACKET == 0, "storage groups of whole packets") };
        let at = At::of::<STORAGE>(row, p);
        let coefficients = unsafe { K::decode::<STORAGE>(at) };
        let values = unsafe { C::values(at) };
        for half in 0..2 {
            let (scale, bias) = coefficients[half];
            for i in 16 * half..16 * half + 16 {
                out[i] = if K::BIASED {
                    scale.mul_add(values[i], bias)
                } else {
                    scale * values[i]
                };
            }
        }
    }

    unsafe fn dot_q8<const R: usize, const M: usize, V: Vectors>(
        _rows: &[Row; R],
        _b: usize,
        _k: usize,
        _x: [&Q8Block; M],
    ) -> [[f32; M]; R] {
        unreachable!("floating codes take no int8 form (`forms::cpu_int8`)")
    }
}

/// One quantized dot of tier `L`'s int8 forms, for the machine line's int8
/// probe (assessment's `line_int8`), so the probe issues the instruction the
/// forms do and the tier's dot paths have one owner: `acc` plus, per 32-bit
/// lane, the four products of unsigned weight bytes and signed activation
/// bytes, by `vpdpbusd` (AVX-512 VNNI's EVEX form, AVX-VNNI's VEX form on a
/// tier without it) where the tier has VNNI, else `maddubs` pairs.
///
/// # Safety
/// The caller is compiled with `L`'s features (an `#[inline(always)]` body
/// below its tier boundary).
#[cfg(target_arch = "x86_64")]
#[cfg(feature = "machine-line")]
#[inline(always)]
pub unsafe fn quantized_dot<L: Isa>(
    acc: std::arch::x86_64::__m256i,
    weights: std::arch::x86_64::__m256i,
    activations: std::arch::x86_64::__m256i,
) -> std::arch::x86_64::__m256i {
    unsafe { packed_x86::dot::<L>(acc, weights, activations) }
}

/// The line's int8 peak on a 512-bit VNNI tier: `vpdpbusd` over whole
/// registers, 64 products an instruction, twice the forms' 256-bit dots.
///
/// # Safety
/// The caller is compiled with `L`'s features, and `L` has AVX-512 VNNI.
#[cfg(target_arch = "x86_64")]
#[cfg(feature = "machine-line")]
#[inline(always)]
pub unsafe fn quantized_dot_wide<L: Isa>(
    acc: std::arch::x86_64::__m512i,
    weights: std::arch::x86_64::__m512i,
    activations: std::arch::x86_64::__m512i,
) -> std::arch::x86_64::__m512i {
    debug_assert!(L::TIER.has("avx512vnni"));
    unsafe { std::arch::x86_64::_mm512_dpbusd_epi32(acc, weights, activations) }
}

/// One quantized dot of the NeonDot tier's int8 forms (`sdot`), as
/// [`quantized_dot`] is x86's.
///
/// # Safety
/// The caller is compiled with the NeonDot tier's features.
#[cfg(target_arch = "aarch64")]
#[cfg(feature = "machine-line")]
#[inline(always)]
pub unsafe fn quantized_dot_neon(
    acc: std::arch::aarch64::int32x4_t,
    a: std::arch::aarch64::int8x16_t,
    b: std::arch::aarch64::int8x16_t,
) -> std::arch::aarch64::int32x4_t {
    unsafe { packed_neon::sdot(acc, a, b) }
}

/// Register-resident packet decodes and dots on NEON. A packet's values are
/// formed in two vectors in element order and never leave registers. The
/// dots use `sdot`, which only the NeonDot tier's boundaries reach.
#[cfg(target_arch = "aarch64")]
mod packed_neon {
    use std::arch::aarch64::*;

    /// `acc` plus the four-way signed byte dot products of `a` and `b`. It
    /// inlines into the NeonDot tier's boundaries, whose features include
    /// `dotprod`.
    ///
    /// # Safety
    /// The CPU has `dotprod`.
    #[inline]
    #[target_feature(enable = "dotprod")]
    pub(super) unsafe fn sdot(mut acc: int32x4_t, a: int8x16_t, b: int8x16_t) -> int32x4_t {
        // Rust's dot-product intrinsic is not stable yet.
        unsafe {
            std::arch::asm!(
                "sdot {acc:v}.4s, {a:v}.16b, {b:v}.16b",
                acc = inout(vreg) acc,
                a = in(vreg) a,
                b = in(vreg) b,
                options(pure, nomem, nostack, preserves_flags),
            );
        }
        acc
    }

    /// `acc` plus, in lane `j`, the signed byte dot of `a`'s bytes
    /// `4j..4j + 4` with `b`'s bytes `4 LANE..4 LANE + 4`.
    ///
    /// # Safety
    /// The CPU has `dotprod`.
    #[inline]
    #[target_feature(enable = "dotprod")]
    pub(super) unsafe fn sdot_lane<const LANE: i32>(
        mut acc: int32x4_t,
        a: int8x16_t,
        b: int8x16_t,
    ) -> int32x4_t {
        unsafe {
            std::arch::asm!(
                "sdot {acc:v}.4s, {a:v}.16b, {b:v}.4b[{lane}]",
                acc = inout(vreg) acc,
                a = in(vreg) a,
                b = in(vreg) b,
                lane = const LANE,
                options(pure, nomem, nostack, preserves_flags),
            );
        }
        acc
    }

    /// The packet's raw `BITS`-bit codes of one row of a rows8 panel from
    /// `units` (`CodeBytes::Panel`), in element order: its 4 bytes of each
    /// of the packet's units gathered, each code's byte selected and
    /// shifted down to its slot.
    #[inline(always)]
    pub(super) unsafe fn panel<const BITS: u32>(units: *const u8) -> [int8x16_t; 2] {
        // Loops, not array helpers: those compile out of line, a call a
        // packet and row.
        let mut words = [0u32; 8];
        for (u, word) in words.iter_mut().enumerate().take(BITS as usize) {
            *word = unsafe { super::read(units.add(32 * u)) };
        }
        unsafe {
            if BITS == 8 {
                return [
                    vreinterpretq_s8_u32(vld1q_u32(words.as_ptr())),
                    vreinterpretq_s8_u32(vld1q_u32(words[4..].as_ptr())),
                ];
            }
            let (index, shift) = const {
                // (A plane of no bits, a component's absent high plane, is never read.)
                let (chunk, mut index, mut shift) = (32 / if BITS == 0 { 8 } else { BITS as usize }, [[0u8; 16]; 2], [[0i8; 16]; 2]);
                let mut c = 0;
                while c < 32 {
                    index[c / 16][c % 16] = (c / chunk * 4 + c % 4) as u8;
                    shift[c / 16][c % 16] = -((c % chunk / 4) as i8 * BITS as i8);
                    c += 1;
                }
                (index, shift)
            };
            let gathered = vld1q_u8(words.as_ptr().cast());
            let mask = vdupq_n_u8(((1u16 << BITS) - 1) as u8);
            let mut halves = [vdupq_n_s8(0); 2];
            for (half, out) in halves.iter_mut().enumerate() {
                let bytes = vqtbl1q_u8(gathered, vld1q_u8(index[half].as_ptr()));
                *out = vreinterpretq_s8_u8(vandq_u8(vshlq_u8(bytes, vld1q_s8(shift[half].as_ptr())), mask));
            }
            halves
        }
    }

    /// Step `step` (four columns) of a rows8 panel span of `BITS`-bit codes
    /// from `units`, rows `4 quad .. 4 quad + 4`: one unit's half, its slot's
    /// codes shifted down.
    #[inline(always)]
    pub(super) unsafe fn panel_step(units: *const u8, step: usize, quad: usize, bits: u32) -> uint8x16_t {
        let slots = 8 / bits as usize;
        unsafe {
            let bytes = vld1q_u8(units.add(32 * (step / slots) + 16 * quad));
            if bits == 8 {
                return bytes;
            }
            let shift = vdupq_n_s8(-((step % slots) as i8 * bits as i8));
            vandq_u8(vshlq_u8(bytes, shift), vdupq_n_u8(((1u16 << bits) - 1) as u8))
        }
    }

    /// Four rows' sixteen bytes as four vectors of their 4-byte chunks:
    /// vector `c` holds chunk `c` of rows 0..4, a row a lane.
    #[inline(always)]
    pub(super) fn chunks4(rows: [int8x16_t; 4]) -> [int8x16_t; 4] {
        unsafe {
            let [a, b, c, d] = rows.map(|row| vreinterpretq_s32_s8(row));
            let (ab0, ab1) = (vtrn1q_s32(a, b), vtrn2q_s32(a, b));
            let (cd0, cd1) = (vtrn1q_s32(c, d), vtrn2q_s32(c, d));
            let pair = |x: int32x4_t, y: int32x4_t, high: bool| {
                let (x, y) = (vreinterpretq_s64_s32(x), vreinterpretq_s64_s32(y));
                vreinterpretq_s8_s64(if high { vtrn2q_s64(x, y) } else { vtrn1q_s64(x, y) })
            };
            [pair(ab0, cd0, false), pair(ab1, cd1, false), pair(ab0, cd0, true), pair(ab1, cd1, true)]
        }
    }

    /// The 32-value dot of two signed packets.
    ///
    /// # Safety
    /// Both pointers address 32 values, and the CPU has `dotprod`.
    #[inline(always)]
    pub(super) unsafe fn dot32(weights: *const i8, activations: *const i8) -> i32 {
        unsafe {
            let zero = vdupq_n_s32(0);
            let first = sdot(zero, vld1q_s8(weights), vld1q_s8(activations));
            vaddvq_s32(sdot(
                first,
                vld1q_s8(weights.add(16)),
                vld1q_s8(activations.add(16)),
            ))
        }
    }

    /// Sixteen values widened to F32.
    #[inline(always)]
    fn widen(values: int8x16_t) -> [float32x4_t; 4] {
        unsafe {
            let (low, high) = (
                vmovl_s8(vget_low_s8(values)),
                vmovl_s8(vget_high_s8(values)),
            );
            [
                vcvtq_f32_s32(vmovl_s16(vget_low_s16(low))),
                vcvtq_f32_s32(vmovl_s16(vget_high_s16(low))),
                vcvtq_f32_s32(vmovl_s16(vget_low_s16(high))),
                vcvtq_f32_s32(vmovl_s16(vget_high_s16(high))),
            ]
        }
    }

    /// `out[i] = scale * values[i] + bias`, fused, for sixteen values.
    ///
    /// # Safety
    /// `out` addresses 16 values.
    #[inline(always)]
    pub(super) unsafe fn affine_store(values: int8x16_t, scale: f32, bias: f32, out: *mut f32) {
        unsafe {
            let (scale, bias) = (vdupq_n_f32(scale), vdupq_n_f32(bias));
            for (quarter, values) in widen(values).into_iter().enumerate() {
                vst1q_f32(out.add(4 * quarter), vfmaq_f32(bias, values, scale));
            }
        }
    }

    /// `out[i] = scale * values[i]` for sixteen values.
    ///
    /// # Safety
    /// `out` addresses 16 values.
    #[inline(always)]
    pub(super) unsafe fn scaled_store(values: int8x16_t, scale: f32, out: *mut f32) {
        unsafe {
            let scale = vdupq_n_f32(scale);
            for (quarter, values) in widen(values).into_iter().enumerate() {
                vst1q_f32(out.add(4 * quarter), vmulq_f32(scale, values));
            }
        }
    }

    /// Bit `i` of `mask` as the value `2^LOW` in byte `i`.
    #[inline(always)]
    fn high_bits<const LOW: u32>(mask: u16) -> uint8x16_t {
        const BITS: [u8; 16] = [1, 2, 4, 8, 16, 32, 64, 128, 1, 2, 4, 8, 16, 32, 64, 128];
        unsafe {
            let bytes = vcombine_u8(vdup_n_u8(mask as u8), vdup_n_u8((mask >> 8) as u8));
            vandq_u8(vtstq_u8(bytes, vld1q_u8(BITS.as_ptr())), vdupq_n_u8(1 << LOW))
        }
    }

    /// A packet's codes of a `LOW`-bit low plane (`4 LOW` bytes) in element
    /// order: code `(8 / LOW) i + j` is bits `LOW j ..` of byte `i`, joined
    /// with its bit of `high` (a 1-bit high plane's packet word) at bit
    /// `LOW`. Codes are at most 31, so they are also signed bytes.
    ///
    /// # Safety
    /// `bytes` addresses the packet's `4 LOW` code bytes.
    #[inline(always)]
    pub(super) unsafe fn codes<const LOW: u32>(bytes: *const u8, high: u32) -> [int8x16_t; 2] {
        const { assert!(matches!(LOW, 1 | 2 | 4), "a 1-, 2- or 4-bit low plane") };
        unsafe {
            if LOW == 4 {
                nibbles(bytes, high)
            } else if LOW == 2 {
                crumbs(bytes, high)
            } else {
                bits(bytes, high)
            }
        }
    }

    /// [`codes`] of a 1-bit low plane: code `i` is bit `i` of the packet's
    /// word.
    #[inline(always)]
    unsafe fn bits(bytes: *const u8, high: u32) -> [int8x16_t; 2] {
        unsafe {
            let word = super::read::<u32>(bytes);
            let mut first = high_bits::<0>(word as u16);
            let mut second = high_bits::<0>((word >> 16) as u16);
            if high != 0 {
                first = vorrq_u8(first, high_bits::<1>(high as u16));
                second = vorrq_u8(second, high_bits::<1>((high >> 16) as u16));
            }
            [vreinterpretq_s8_u8(first), vreinterpretq_s8_u8(second)]
        }
    }

    /// [`codes`] of a 4-bit low plane: codes 2i and 2i + 1 are the low and
    /// high nibble of byte i.
    #[inline(always)]
    unsafe fn nibbles(bytes: *const u8, high: u32) -> [int8x16_t; 2] {
        unsafe {
            let packed = vld1q_u8(bytes);
            let low = vandq_u8(packed, vdupq_n_u8(15));
            let upper = vshrq_n_u8(packed, 4);
            let mut first = vzip1q_u8(low, upper);
            let mut second = vzip2q_u8(low, upper);
            if high != 0 {
                first = vorrq_u8(first, high_bits::<4>(high as u16));
                second = vorrq_u8(second, high_bits::<4>((high >> 16) as u16));
            }
            [vreinterpretq_s8_u8(first), vreinterpretq_s8_u8(second)]
        }
    }

    /// [`codes`] of a 2-bit low plane: codes 4i .. 4i + 4 are bits 0, 2, 4
    /// and 6 of byte i, each byte spread to its four codes' lanes and
    /// shifted right by the code's bit.
    #[inline(always)]
    unsafe fn crumbs(bytes: *const u8, high: u32) -> [int8x16_t; 2] {
        const SPREAD: [[u8; 16]; 2] = [
            [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3],
            [4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6, 7, 7, 7, 7],
        ];
        const SHIFTS: [i8; 16] = [0, -2, -4, -6, 0, -2, -4, -6, 0, -2, -4, -6, 0, -2, -4, -6];
        unsafe {
            let packed = vreinterpretq_u8_u64(vld1q_dup_u64(bytes.cast()));
            let shifts = vld1q_s8(SHIFTS.as_ptr());
            let three = vdupq_n_u8(3);
            let [mut first, mut second] = SPREAD.map(|spread| {
                let spread = vqtbl1q_u8(packed, vld1q_u8(spread.as_ptr()));
                vandq_u8(vshlq_u8(spread, shifts), three)
            });
            if high != 0 {
                first = vorrq_u8(first, high_bits::<2>(high as u16));
                second = vorrq_u8(second, high_bits::<2>((high >> 16) as u16));
            }
            [vreinterpretq_s8_u8(first), vreinterpretq_s8_u8(second)]
        }
    }

    /// A packet's codes in element order under a 2-bit high plane: codes 2i
    /// and 2i + 1 join the low and high nibble of byte i with bits 0-1 and
    /// 2-3 of the high plane's nibble i. Codes are at most 63.
    ///
    /// # Safety
    /// `codes` addresses the packet's 16 code bytes and `high` its 8 high
    /// bytes.
    #[inline(always)]
    pub(super) unsafe fn nibbles2(codes: *const u8, high: *const u8) -> [int8x16_t; 2] {
        unsafe {
            let packed = vld1q_u8(codes);
            let high = vld1_u8(high);
            let (low_nibbles, high_nibbles) = (vand_u8(high, vdup_n_u8(15)), vshr_n_u8(high, 4));
            let nibbles = vcombine_u8(
                vzip1_u8(low_nibbles, high_nibbles),
                vzip2_u8(low_nibbles, high_nibbles),
            );
            let even = vorrq_u8(
                vandq_u8(packed, vdupq_n_u8(15)),
                vshlq_n_u8(vandq_u8(nibbles, vdupq_n_u8(3)), 4),
            );
            let odd = vorrq_u8(vshrq_n_u8(packed, 4), vshlq_n_u8(vshrq_n_u8(nibbles, 2), 4));
            [
                vreinterpretq_s8_u8(vzip1q_u8(even, odd)),
                vreinterpretq_s8_u8(vzip2q_u8(even, odd)),
            ]
        }
    }

    use super::{lane_order, Lanes};

    /// Each step's columns in element order, in a table of the four lanes'
    /// bytes (`lane_order`): of the whole k-block for 4-bit codes (a byte a
    /// slot pair), a 1-bit plane (a byte a step) and a 2-bit plane (two
    /// bytes a step), each before its packet's half's steps; of the
    /// packet's steps for 8-bit codes (two bytes a slot pair, before the
    /// row's slot).
    const STEP_ORDER: [u8; 16] = lane_order(4, 0, 1, 1)[0];
    const BYTES: [[u8; 16]; 2] = lane_order(16, 8, 2, 1);
    const ONE_BIT: [[u8; 16]; 2] = lane_order(4, 1, 0, 1);
    const TWO_BITS: [[u8; 16]; 2] = lane_order(8, 2, 1, 2);
    /// Per column, its slot pair's bit of a 1-bit step byte (the pair's
    /// first slot), and its shift in its byte of a 2-bit step word.
    const ONE_BIT_MASKS: [u8; 16] = {
        let mut masks = [0u8; 16];
        let mut k = 0;
        while k < 16 {
            masks[k] = 1 << (2 * super::lane_slot(k).1);
            k += 1;
        }
        masks
    };
    const TWO_BIT_SHIFTS: [i8; 16] = {
        let mut shifts = [0i8; 16];
        let mut k = 0;
        while k < 16 {
            shifts[k] = -4 * (super::lane_slot(k).1 % 2) as i8;
            k += 1;
        }
        shifts
    };

    /// Each step's table indices `order`, at the packet's half (`half`
    /// steps of `step` bytes further).
    #[inline(always)]
    unsafe fn at_half(order: &[[u8; 16]; 2], half: usize, step: u8) -> [uint8x16_t; 2] {
        unsafe {
            let offset = vdupq_n_u8(2 * half as u8 * step);
            [
                vaddq_u8(vld1q_u8(order[0].as_ptr()), offset),
                vaddq_u8(vld1q_u8(order[1].as_ptr()), offset),
            ]
        }
    }

    /// The row's 4-bit codes of an `mma16` codes plane in element order.
    #[inline(always)]
    pub(super) unsafe fn lane_nibbles(lanes: Lanes) -> [int8x16_t; 2] {
        unsafe {
            // Each lane's two steps of the packet, then each step's four
            // lanes (`[t][b]`), in element order.
            let steps = |pair: *const u8| {
                vcombine_u8(vld1_u8(pair.add(8 * lanes.half)), vld1_u8(pair.add(lanes.unit + 8 * lanes.half)))
            };
            let (first, second) = (
                vreinterpretq_u32_u8(steps(lanes.pairs[0])),
                vreinterpretq_u32_u8(steps(lanes.pairs[1])),
            );
            let order = vld1q_u8(STEP_ORDER.as_ptr());
            let bytes = [
                vqtbl1q_u8(vreinterpretq_u8_u32(vuzp1q_u32(first, second)), order),
                vqtbl1q_u8(vreinterpretq_u8_u32(vuzp2q_u32(first, second)), order),
            ];
            let nibble = |bytes: uint8x16_t| {
                if lanes.upper {
                    vshrq_n_u8(bytes, 4)
                } else {
                    vandq_u8(bytes, vdupq_n_u8(15))
                }
            };
            [vreinterpretq_s8_u8(nibble(bytes[0])), vreinterpretq_s8_u8(nibble(bytes[1]))]
        }
    }

    /// The four steps of an `mma16` 4-bit k-block of a row pair, each in
    /// element order, both rows' codes in each byte (row `g` low, `g + 8`
    /// high): the four lanes whole, their four steps transposed.
    #[inline(always)]
    unsafe fn block_steps(lanes: Lanes) -> [uint8x16_t; 4] {
        unsafe {
            let (first, second) = (vld1q_u8_x2(lanes.pairs[0]), vld1q_u8_x2(lanes.pairs[1]));
            let words = |lane: uint8x16_t| vreinterpretq_u32_u8(lane);
            let (l0, l1, l2, l3) = (words(first.0), words(first.1), words(second.0), words(second.1));
            let (even01, odd01) = (vtrn1q_u32(l0, l1), vtrn2q_u32(l0, l1));
            let (even23, odd23) = (vtrn1q_u32(l2, l3), vtrn2q_u32(l2, l3));
            let quads = |words: uint32x4_t| vreinterpretq_u64_u32(words);
            let order = vld1q_u8(STEP_ORDER.as_ptr());
            let step = |low: uint32x4_t, high: uint32x4_t, second: bool| {
                let words = if second {
                    vtrn2q_u64(quads(low), quads(high))
                } else {
                    vtrn1q_u64(quads(low), quads(high))
                };
                vqtbl1q_u8(vreinterpretq_u8_u64(words), order)
            };
            [
                step(even01, even23, false),
                step(odd01, odd23, false),
                step(even01, even23, true),
                step(odd01, odd23, true),
            ]
        }
    }

    /// One row's nibbles of a step's bytes: the low (`g`) or the high
    /// (`g + 8`).
    #[inline(always)]
    fn nibbles_of(bytes: uint8x16_t, upper: bool) -> int8x16_t {
        unsafe {
            vreinterpretq_s8_u8(if upper {
                vshrq_n_u8(bytes, 4)
            } else {
                vandq_u8(bytes, vdupq_n_u8(15))
            })
        }
    }

    /// The row's 4-bit codes of both packets of an `mma16` k-block, each in
    /// element order.
    #[inline(always)]
    pub(super) unsafe fn block_nibbles(lanes: Lanes) -> [[int8x16_t; 2]; 2] {
        let [s0, s1, s2, s3] = unsafe { block_steps(lanes) };
        let row = |step| nibbles_of(step, lanes.upper);
        [[row(s0), row(s1)], [row(s2), row(s3)]]
    }

    /// [`block_nibbles`] of both rows of the pair (`g`, then `g + 8`), the
    /// k-block's bytes read once.
    #[inline(always)]
    pub(super) unsafe fn block_partners(lanes: Lanes) -> [[[int8x16_t; 2]; 2]; 2] {
        let [s0, s1, s2, s3] = unsafe { block_steps(lanes) };
        [false, true].map(|upper| {
            let row = |step| nibbles_of(step, upper);
            [[row(s0), row(s1)], [row(s2), row(s3)]]
        })
    }

    /// The row's `bits`-bit codes (1 or 2) of an `mma16` codes or high plane
    /// in element order.
    #[inline(always)]
    unsafe fn lane_bits(lanes: Lanes, bits: u32) -> [uint8x16_t; 2] {
        unsafe {
            if bits == 1 {
                let table = vcombine_u8(vld1_u8(lanes.pairs[0]), vld1_u8(lanes.pairs[1]));
                let masks = vshlq_u8(vld1q_u8(ONE_BIT_MASKS.as_ptr()), vdupq_n_s8(lanes.upper as i8));
                let one = vdupq_n_u8(1);
                let order = at_half(&ONE_BIT, lanes.half, 1);
                let bit = |order: uint8x16_t| vandq_u8(vtstq_u8(vqtbl1q_u8(table, order), masks), one);
                [bit(order[0]), bit(order[1])]
            } else {
                let table = uint8x16x2_t(vld1q_u8(lanes.pairs[0]), vld1q_u8(lanes.pairs[1]));
                let shifts = vsubq_s8(vld1q_s8(TWO_BIT_SHIFTS.as_ptr()), vdupq_n_s8(2 * lanes.upper as i8));
                let three = vdupq_n_u8(3);
                let order = at_half(&TWO_BITS, lanes.half, 2);
                let pair = |order: uint8x16_t| vandq_u8(vshlq_u8(vqtbl2q_u8(table, order), shifts), three);
                [pair(order[0]), pair(order[1])]
            }
        }
    }

    /// The row's `LOW`-bit codes (1, 2 or 4) of an `mma16` codes plane in
    /// element order.
    #[inline(always)]
    pub(super) unsafe fn lane_codes<const LOW: u32>(lanes: Lanes) -> [int8x16_t; 2] {
        match LOW {
            4 => unsafe { lane_nibbles(lanes) },
            _ => unsafe { lane_bits(lanes, LOW) }.map(|codes| unsafe { vreinterpretq_s8_u8(codes) }),
        }
    }

    /// `codes` joined at bit `LOW` with the row's `bits`-bit codes (1 or 2)
    /// of an `mma16` high plane.
    #[inline(always)]
    pub(super) unsafe fn with_high<const LOW: u32>(
        codes: [int8x16_t; 2],
        lanes: Lanes,
        bits: u32,
    ) -> [int8x16_t; 2] {
        unsafe {
            let high = lane_bits(lanes, bits);
            let shift = vdupq_n_s8(LOW as i8);
            [0, 1].map(|half| {
                vreinterpretq_s8_u8(vorrq_u8(
                    vreinterpretq_u8_s8(codes[half]),
                    vshlq_u8(high[half], shift),
                ))
            })
        }
    }

    /// The row's 8-bit codes of an `mma16` codes plane in element order.
    #[inline(always)]
    pub(super) unsafe fn lane_bytes(lanes: Lanes) -> [int8x16_t; 2] {
        unsafe {
            let table = uint8x16x4_t(
                vld1q_u8(lanes.lane(0)),
                vld1q_u8(lanes.lane(1)),
                vld1q_u8(lanes.lane(2)),
                vld1q_u8(lanes.lane(3)),
            );
            let upper = vdupq_n_u8(lanes.upper as u8);
            [
                vreinterpretq_s8_u8(vqtbl4q_u8(table, vaddq_u8(vld1q_u8(BYTES[0].as_ptr()), upper))),
                vreinterpretq_s8_u8(vqtbl4q_u8(table, vaddq_u8(vld1q_u8(BYTES[1].as_ptr()), upper))),
            ]
        }
    }
}

/// Per step `s` of a packet of an `mma16` codes plane, each column's index
/// in the lanes' bytes of the packet's steps gathered lane by lane (`lane`
/// bytes a lane, `step` a step): its lane's, its step's, then its slot
/// pair `b`'s `b * numerator / denominator`.
#[cfg_attr(
    not(any(
        target_arch = "aarch64",
        target_arch = "x86_64"
    )),
    allow(dead_code)
)]
const fn lane_order(lane: usize, step: usize, numerator: usize, denominator: usize) -> [[u8; 16]; 2] {
    let mut order = [[0u8; 16]; 2];
    let mut s = 0;
    while s < 2 {
        let mut k = 0;
        while k < 16 {
            let (t, b) = lane_slot(k);
            order[s][k] = (t * lane + s * step + b * numerator / denominator) as u8;
            k += 1;
        }
        s += 1;
    }
    order
}

/// Register-resident packet dots for the x86 tiers with 256-bit integer
/// vectors: mode 1 (AVX2) and mode 2 (AVX-512 VNNI on 256-bit registers).
/// A packet's 32 values form one vector in element order and its products
/// stay in eight `i32` lanes until the caller's block ends, so integer sums
/// are exact and independent of the lane order. Only components of a tier
/// whose features cover the mode reach these bodies; they inline into them.
///
/// Weight bytes passed as unsigned are at most 128 (255 to mode 2's `dot`)
/// and activations are quantization codes in `[-127, 127]`, so the AVX2 pair
/// sums of 16-bit products never saturate (`2 * 128 * 127 < 2^15`).
#[cfg(target_arch = "x86_64")]
mod packed_x86 {
    use super::{Codes, Vectors};
    use std::arch::x86_64::*;

    /// x86's products of unsigned weight bytes and signed activation bytes
    /// (at least -127 each: `quant::quantize`).
    #[derive(Clone, Copy)]
    pub(super) enum Products {
        /// `maddubs`: pairs of products into 16-bit lanes, exact for bytes
        /// up to 128 (`2 · 128 · 127 ≤ 32767`).
        Bytes16,
        /// VNNI's `vpdpbusd`: four products into 32-bit lanes, of any byte.
        Dots32,
    }

    impl Products {
        /// The products of tier `V`'s dots.
        pub(super) const fn of<V: Vectors>() -> Self {
            if V::QUANTIZED_DOTS == 2 { Self::Dots32 } else { Self::Bytes16 }
        }

        /// The largest byte the products take.
        const fn greatest(self) -> i32 {
            match self {
                Self::Bytes16 => 128,
                Self::Dots32 => 255,
            }
        }
    }

    /// How every x86 dot takes codes `C` into `products`: the value it
    /// takes them less as unsigned bytes, their least, when their range fits
    /// the products' bytes (each dot then adding back `least Σ a`); `None`
    /// for the sign form, their magnitudes against the activations signed by
    /// the codes.
    pub(super) const fn shift<C: Codes>(products: Products) -> Option<i32> {
        if C::GREATEST - C::LEAST <= products.greatest() { Some(C::LEAST) } else { None }
    }

    /// 32 bytes at `bytes`.
    #[inline(always)]
    pub(super) unsafe fn load(bytes: *const i8) -> __m256i {
        unsafe { _mm256_loadu_si256(bytes.cast()) }
    }

    /// The packet's raw `BITS`-bit codes of one row of a rows8 panel from
    /// `units` (`CodeBytes::Panel`), in element order: its 4 bytes of each
    /// of the packet's units gathered (in both lanes), each code's byte
    /// selected and its four-code lane shifted down to their slot.
    #[inline(always)]
    pub(super) unsafe fn panel<const BITS: u32>(units: *const u8) -> __m256i {
        // Loops, not array helpers: those compile out of line, a call a
        // packet and row.
        let mut words = [0i32; 8];
        for (u, word) in words.iter_mut().enumerate().take(BITS as usize) {
            *word = unsafe { super::read(units.add(32 * u)) };
        }
        unsafe {
            if BITS == 8 {
                return _mm256_loadu_si256(words.as_ptr().cast());
            }
            let (index, shift) = const {
                // (A plane of no bits, a component's absent high plane, is never read.)
                let (chunk, mut index, mut shift) = (32 / if BITS == 0 { 8 } else { BITS as usize }, [0u8; 32], [0i32; 8]);
                let mut c = 0;
                while c < 32 {
                    index[c] = (c / chunk * 4 + c % 4) as u8;
                    shift[c / 4] = (c % chunk / 4) as i32 * BITS as i32;
                    c += 1;
                }
                (index, shift)
            };
            let gathered = _mm256_broadcastsi128_si256(_mm_loadu_si128(words.as_ptr().cast()));
            let bytes = _mm256_shuffle_epi8(gathered, _mm256_loadu_si256(index.as_ptr().cast()));
            let codes = _mm256_srlv_epi32(bytes, _mm256_loadu_si256(shift.as_ptr().cast()));
            _mm256_and_si256(codes, _mm256_set1_epi8(((1u16 << BITS) - 1) as u8 as i8))
        }
    }

    /// Step `step` (four columns) of a rows8 panel span of `BITS`-bit codes
    /// from `units`, the tile's eight rows: one unit, its slot's codes
    /// shifted down.
    #[inline(always)]
    pub(super) unsafe fn panel_step(units: *const u8, step: usize, bits: u32) -> __m256i {
        let slots = 8 / bits as usize;
        unsafe {
            let bytes = _mm256_loadu_si256(units.add(32 * (step / slots)).cast());
            if bits == 8 {
                return bytes;
            }
            let codes = _mm256_srl_epi16(bytes, _mm_cvtsi32_si128((step % slots) as i32 * bits as i32));
            _mm256_and_si256(codes, _mm256_set1_epi8(((1u16 << bits) - 1) as u8 as i8))
        }
    }

    /// `acc` plus, per 32-bit lane, the four products of unsigned weight
    /// bytes `weights` and signed activation bytes `activations`.
    #[inline(always)]
    pub(super) unsafe fn dot<V: crate::isa::Isa>(
        acc: __m256i,
        weights: __m256i,
        activations: __m256i,
    ) -> __m256i {
        unsafe {
            if V::QUANTIZED_DOTS == 2 {
                dpbusd::<V>(acc, weights, activations)
            } else {
                let pairs = _mm256_maddubs_epi16(weights, activations);
                _mm256_add_epi32(acc, _mm256_madd_epi16(pairs, _mm256_set1_epi16(1)))
            }
        }
    }

    /// VNNI's `vpdpbusd` on a VNNI tier `V`: `acc` plus, per 32-bit lane,
    /// the four products of unsigned bytes `weights` and signed bytes
    /// `activations`. AVX-512 VNNI's EVEX form on a tier with it, else
    /// AVX-VNNI's VEX form (an AVX-512 VNNI part need not have AVX-VNNI).
    #[inline(always)]
    pub(super) unsafe fn dpbusd<V: crate::isa::Isa>(acc: __m256i, weights: __m256i, activations: __m256i) -> __m256i {
        unsafe {
            if V::AVX512_VNNI {
                _mm256_dpbusd_epi32(acc, weights, activations)
            } else {
                _mm256_dpbusd_avx_epi32(acc, weights, activations)
            }
        }
    }

    /// VNNI's `vpdpwssd` on a VNNI tier `V`, as [`dpbusd`]: `acc` plus, per
    /// 32-bit lane, the products of the 16-bit halves of `a` and `b`.
    #[inline(always)]
    pub(super) unsafe fn dpwssd<V: Vectors>(acc: __m256i, a: __m256i, b: __m256i) -> __m256i {
        unsafe {
            if V::AVX512_VNNI {
                _mm256_dpwssd_epi32(acc, a, b)
            } else {
                _mm256_dpwssd_avx_epi32(acc, a, b)
            }
        }
    }

    /// `acc` plus the four-way products of signed weights and activations:
    /// `|w| * (a * sign(w))` is `w * a` exactly, with `|w|` at most 128.
    #[inline(always)]
    pub(super) unsafe fn dot_signed<V: Vectors>(
        acc: __m256i,
        weights: __m256i,
        activations: __m256i,
    ) -> __m256i {
        unsafe {
            dot::<V>(
                acc,
                _mm256_abs_epi8(weights),
                _mm256_sign_epi8(activations, weights),
            )
        }
    }

    /// The products of one packet's codes `weights`, as [`shift`] takes
    /// them (less `least` with `Some(least)`, else in the sign form), with
    /// packet `local` of the activation block `x`.
    #[inline(always)]
    pub(super) unsafe fn product<V: Vectors>(
        weights: __m256i,
        x: &crate::quant::Q8Block,
        local: usize,
        shift: Option<i32>,
    ) -> __m256i {
        unsafe {
            let activations = load(x.codes.as_ptr().add(super::PACKET * local));
            match shift {
                Some(least) => dot::<V>(_mm256_setzero_si256(), operands(weights, least), activations),
                None => dot_signed::<V>(_mm256_setzero_si256(), weights, activations),
            }
        }
    }

    /// Codes `weights` less `least`, as unsigned bytes.
    #[inline(always)]
    pub(super) unsafe fn operands(weights: __m256i, least: i32) -> __m256i {
        unsafe { if least == 0 { weights } else { _mm256_sub_epi8(weights, _mm256_set1_epi8(least as i8)) } }
    }

    /// `acc` plus the pair products of unsigned weights and activations,
    /// each pair multiplied by its 16-bit lane of `scales` (at most 128 in
    /// magnitude, so a lane stays far inside `i32`).
    #[inline(always)]
    pub(super) unsafe fn dot_scaled(
        acc: __m256i,
        weights: __m256i,
        activations: __m256i,
        scales: __m256i,
    ) -> __m256i {
        unsafe {
            let pairs = _mm256_maddubs_epi16(weights, activations);
            _mm256_add_epi32(acc, _mm256_madd_epi16(pairs, scales))
        }
    }

    /// The sum of the eight lanes.
    #[inline(always)]
    pub(super) unsafe fn sum(lanes: __m256i) -> i32 {
        unsafe {
            let half = _mm_add_epi32(
                _mm256_castsi256_si128(lanes),
                _mm256_extracti128_si256::<1>(lanes),
            );
            let pair = _mm_add_epi32(half, _mm_shuffle_epi32::<0b01_00_11_10>(half));
            _mm_cvtsi128_si32(_mm_add_epi32(
                pair,
                _mm_shuffle_epi32::<0b10_11_00_01>(pair),
            ))
        }
    }

    /// Per 128-bit half of four vectors: `[v0 low, v1 low, v2 low, v3 low |
    /// v0 high, v1 high, v2 high, v3 high]`, each the sum of four lanes.
    #[inline(always)]
    pub(super) unsafe fn halves4(v: [__m256i; 4]) -> __m256i {
        unsafe { _mm256_hadd_epi32(_mm256_hadd_epi32(v[0], v[1]), _mm256_hadd_epi32(v[2], v[3])) }
    }

    /// The sums of the low and of the high halves of eight vectors, in
    /// vector order.
    #[inline(always)]
    pub(super) unsafe fn halves8(v: [__m256i; 8]) -> (__m256i, __m256i) {
        unsafe {
            let first = halves4([v[0], v[1], v[2], v[3]]);
            let second = halves4([v[4], v[5], v[6], v[7]]);
            (
                _mm256_permute2x128_si256::<0x20>(first, second),
                _mm256_permute2x128_si256::<0x31>(first, second),
            )
        }
    }

    /// A packet's `4 LOW` bytes of `LOW`-bit codes as 32 codes in element
    /// order: code `(8 / LOW) i + j` is bits `LOW j ..` of byte `i`.
    #[inline(always)]
    pub(super) unsafe fn codes<const LOW: u32>(bytes: *const u8) -> __m256i {
        const { assert!(matches!(LOW, 1 | 2 | 4), "a 1-, 2- or 4-bit low plane") };
        unsafe {
            if LOW == 1 {
                // Code i is bit i of the packet's word.
                high_bits::<0>(super::read(bytes))
            } else if LOW == 4 {
                let packed = _mm_loadu_si128(bytes.cast());
                let mask = _mm_set1_epi8(15);
                let low = _mm_and_si128(packed, mask);
                let high = _mm_and_si128(_mm_srli_epi16::<4>(packed), mask);
                _mm256_set_m128i(_mm_unpackhi_epi8(low, high), _mm_unpacklo_epi8(low, high))
            } else {
                // Codes 4i..4i + 4 of byte i: its pairs of bits at 0, 2, 4
                // and 6, interleaved byte by byte, then pair by pair.
                let packed = _mm_loadl_epi64(bytes.cast());
                let three = _mm_set1_epi8(3);
                let pairs = [
                    _mm_and_si128(packed, three),
                    _mm_and_si128(_mm_srli_epi16::<2>(packed), three),
                    _mm_and_si128(_mm_srli_epi16::<4>(packed), three),
                    _mm_and_si128(_mm_srli_epi16::<6>(packed), three),
                ];
                let (first, second) = (
                    _mm_unpacklo_epi8(pairs[0], pairs[1]),
                    _mm_unpacklo_epi8(pairs[2], pairs[3]),
                );
                _mm256_set_m128i(
                    _mm_unpackhi_epi16(first, second),
                    _mm_unpacklo_epi16(first, second),
                )
            }
        }
    }

    /// Bit `i` of `mask` as the value `2^LOW` in byte `i`.
    #[inline(always)]
    unsafe fn high_bits<const LOW: u32>(mask: u32) -> __m256i {
        unsafe {
            let spread = _mm256_shuffle_epi8(
                _mm256_set1_epi32(mask as i32),
                _mm256_setr_epi8(
                    0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3,
                    3, 3, 3, 3, 3, 3,
                ),
            );
            let bits = _mm256_set1_epi64x(0x8040_2010_0804_0201u64 as i64);
            let set = _mm256_cmpeq_epi8(_mm256_and_si256(spread, bits), bits);
            _mm256_and_si256(set, _mm256_set1_epi8(1 << LOW))
        }
    }

    /// A packet's `LOW`-bit codes ([`codes`]), each joined with its bit of
    /// `high` (a 1-bit high plane's packet word) at bit `LOW`.
    #[inline(always)]
    pub(super) unsafe fn codes1<const LOW: u32>(bytes: *const u8, high: u32) -> __m256i {
        unsafe { _mm256_or_si256(codes::<LOW>(bytes), high_bits::<LOW>(high)) }
    }

    /// A packet's codes under a 2-bit high plane: codes 2i and 2i + 1 join
    /// the low and high nibble of byte i with bits 0-1 and 2-3 of the high
    /// plane's nibble i.
    #[inline(always)]
    pub(super) unsafe fn nibbles2(codes: *const u8, high: *const u8) -> __m256i {
        unsafe {
            let packed = _mm_loadu_si128(codes.cast());
            let high = _mm_loadl_epi64(high.cast());
            let fifteen = _mm_set1_epi8(15);
            let nibbles = _mm_unpacklo_epi8(
                _mm_and_si128(high, fifteen),
                _mm_and_si128(_mm_srli_epi16::<4>(high), fifteen),
            );
            let three = _mm_set1_epi8(3);
            let even = _mm_or_si128(
                _mm_and_si128(packed, fifteen),
                _mm_slli_epi16::<4>(_mm_and_si128(nibbles, three)),
            );
            let odd = _mm_or_si128(
                _mm_and_si128(_mm_srli_epi16::<4>(packed), fifteen),
                _mm_slli_epi16::<4>(_mm_and_si128(_mm_srli_epi16::<2>(nibbles), three)),
            );
            _mm256_set_m128i(_mm_unpackhi_epi8(even, odd), _mm_unpacklo_epi8(even, odd))
        }
    }

    /// The 32 values of a 16-entry table `table` (broadcast to both halves)
    /// at the nibble codes `codes`.
    #[inline(always)]
    pub(super) unsafe fn lookup(table: __m256i, codes: __m256i) -> __m256i {
        unsafe { _mm256_shuffle_epi8(table, codes) }
    }

    /// A 16-entry byte table in both halves of a vector.
    #[inline(always)]
    pub(super) unsafe fn table(values: &[i8; 16]) -> __m256i {
        unsafe { _mm256_broadcastsi128_si256(_mm_loadu_si128(values.as_ptr().cast())) }
    }

    use super::{lane_order, Lanes};

    /// Per half (step `s`), each column's index in a half of the lanes'
    /// step words gathered step by step (`steps`); in a lane's two step
    /// words broadcast (a 1-bit plane), or its four (a 2-bit plane); and in
    /// a lane's two step words of eight bytes, a slot pair's first byte
    /// (8-bit codes: the step's four slots of the row, then the next
    /// step's, `SELECT`).
    const STEPS: [u8; 32] = halves(lane_order(4, 0, 1, 1));
    const ONE_BIT: [u8; 32] = halves(lane_order(2, 1, 0, 1));
    const TWO_BITS: [u8; 32] = halves(lane_order(4, 2, 1, 2));
    const SELECT: [u8; 32] = {
        let mut select = [0x80u8; 32];
        let mut at = 0;
        while at < 8 {
            let (s, b) = (at / 4, at % 4);
            select[at] = (s * 8 + 2 * b) as u8;
            select[16 + at] = select[at];
            at += 1;
        }
        select
    };
    /// Per column, its slot pair's bit of a 1-bit step byte, and whether
    /// its code is in the high nibble of its byte of a 2-bit step word.
    const ONE_BIT_MASKS: [u8; 32] = {
        let mut masks = [0u8; 32];
        let mut at = 0;
        while at < 32 {
            masks[at] = 1 << (2 * super::lane_slot(at % 16).1);
            at += 1;
        }
        masks
    };
    const TWO_BIT_HIGH: [u8; 32] = {
        let mut high = [0u8; 32];
        let mut at = 0;
        while at < 32 {
            high[at] = if super::lane_slot(at % 16).1 % 2 == 1 { 0x80 } else { 0 };
            at += 1;
        }
        high
    };

    /// Two 16-entry orders as the halves of one.
    const fn halves(order: [[u8; 16]; 2]) -> [u8; 32] {
        let mut both = [0u8; 32];
        let mut at = 0;
        while at < 16 {
            both[at] = order[0][at];
            both[16 + at] = order[1][at];
            at += 1;
        }
        both
    }

    #[inline(always)]
    unsafe fn constant(bytes: &[u8; 32]) -> __m256i {
        unsafe { _mm256_loadu_si256(bytes.as_ptr().cast()) }
    }

    /// The packet's steps of lane `t`, as `T`.
    #[inline(always)]
    unsafe fn lane<T: Copy>(lanes: Lanes, t: usize) -> T {
        unsafe { lanes.lane(t).cast::<T>().read_unaligned() }
    }

    /// Step words of four bytes, lane by lane (`[t0 s0, t0 s1, t1 s0, ..]`),
    /// as the packet's columns in element order: step 0's words of every
    /// lane in the low half, step 1's in the high, each half reordered
    /// (`STEPS`).
    #[inline(always)]
    unsafe fn steps(words: __m256i, order: __m256i) -> __m256i {
        unsafe { _mm256_shuffle_epi8(_mm256_permutevar8x32_epi32(words, order), constant(&STEPS)) }
    }

    /// Both packets of an `mma16` 4-bit k-block of a row pair, each in
    /// element order, both rows' codes in each byte (row `g` low, `g + 8`
    /// high): the four lanes whole, each step's words gathered.
    #[inline(always)]
    unsafe fn block_packets(lanes: Lanes) -> [__m256i; 2] {
        unsafe {
            // [t0 | t1] and [t2 | t3], each lane's four step words; then
            // per half of the k-block [t0 s, t2 s, t0 s', t2 s' | t1 ..].
            let first = _mm256_loadu_si256(lanes.pairs[0].cast());
            let second = _mm256_loadu_si256(lanes.pairs[1].cast());
            let order = _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7);
            [
                steps(_mm256_unpacklo_epi32(first, second), order),
                steps(_mm256_unpackhi_epi32(first, second), order),
            ]
        }
    }

    /// One row's nibbles of a packet's bytes: the low (`g`) or the high
    /// (`g + 8`).
    #[inline(always)]
    unsafe fn nibbles_of(bytes: __m256i, upper: bool) -> __m256i {
        unsafe {
            let count = _mm_cvtsi32_si128(4 * upper as i32);
            _mm256_and_si256(_mm256_srl_epi16(bytes, count), _mm256_set1_epi8(15))
        }
    }

    /// The row's 4-bit codes of both packets of an `mma16` k-block, each in
    /// element order.
    #[inline(always)]
    pub(super) unsafe fn block_nibbles(lanes: Lanes) -> [__m256i; 2] {
        unsafe {
            let [first, second] = block_packets(lanes);
            [nibbles_of(first, lanes.upper), nibbles_of(second, lanes.upper)]
        }
    }

    /// [`block_nibbles`] of both rows of the pair (`g`, then `g + 8`), the
    /// k-block's bytes read once.
    #[inline(always)]
    pub(super) unsafe fn block_partners(lanes: Lanes) -> [[__m256i; 2]; 2] {
        unsafe {
            let [first, second] = block_packets(lanes);
            [
                [nibbles_of(first, false), nibbles_of(second, false)],
                [nibbles_of(first, true), nibbles_of(second, true)],
            ]
        }
    }

    /// The row's 4-bit codes of an `mma16` codes plane in element order.
    #[inline(always)]
    pub(super) unsafe fn lane_nibbles(lanes: Lanes) -> __m256i {
        unsafe {
            // Lanes 0 and 1, and 2 and 3, whole; each lane's half of its
            // steps a quadword: [t0, t2 | t1, t3].
            let first = _mm256_loadu_si256(lanes.pairs[0].cast());
            let second = _mm256_loadu_si256(lanes.pairs[1].cast());
            let words = if lanes.half == 0 {
                _mm256_unpacklo_epi64(first, second)
            } else {
                _mm256_unpackhi_epi64(first, second)
            };
            let shifted = _mm256_srl_epi16(words, _mm_cvtsi32_si128(4 * lanes.upper as i32));
            let nibbles = _mm256_and_si256(shifted, _mm256_set1_epi8(15));
            steps(nibbles, _mm256_setr_epi32(0, 4, 2, 6, 1, 5, 3, 7))
        }
    }

    /// The row's `bits`-bit codes (1 or 2) of an `mma16` codes or high plane
    /// in element order.
    #[inline(always)]
    unsafe fn lane_bits(lanes: Lanes, bits: u32) -> __m256i {
        unsafe {
            if bits == 1 {
                let word = u64::from(lane::<u16>(lanes, 0))
                    | (u64::from(lane::<u16>(lanes, 1)) << 16)
                    | (u64::from(lane::<u16>(lanes, 2)) << 32)
                    | (u64::from(lane::<u16>(lanes, 3)) << 48);
                let bytes = _mm256_shuffle_epi8(_mm256_set1_epi64x(word as i64), constant(&ONE_BIT));
                let base = constant(&ONE_BIT_MASKS);
                let masks = if lanes.upper { _mm256_add_epi8(base, base) } else { base };
                let set = _mm256_cmpeq_epi8(_mm256_and_si256(bytes, masks), masks);
                _mm256_and_si256(set, _mm256_set1_epi8(1))
            } else {
                let words = _mm_setr_epi32(
                    lane::<i32>(lanes, 0),
                    lane::<i32>(lanes, 1),
                    lane::<i32>(lanes, 2),
                    lane::<i32>(lanes, 3),
                );
                let bytes = _mm256_shuffle_epi8(_mm256_broadcastsi128_si256(words), constant(&TWO_BITS));
                let shift = 2 * lanes.upper as i32;
                let low = _mm256_srl_epi16(bytes, _mm_cvtsi32_si128(shift));
                let high = _mm256_srl_epi16(bytes, _mm_cvtsi32_si128(shift + 4));
                let picked = _mm256_blendv_epi8(low, high, constant(&TWO_BIT_HIGH));
                _mm256_and_si256(picked, _mm256_set1_epi8(3))
            }
        }
    }

    /// The row's `LOW`-bit codes (1, 2 or 4) of an `mma16` codes plane in
    /// element order.
    #[inline(always)]
    pub(super) unsafe fn lane_codes<const LOW: u32>(lanes: Lanes) -> __m256i {
        match LOW {
            4 => unsafe { lane_nibbles(lanes) },
            _ => unsafe { lane_bits(lanes, LOW) },
        }
    }

    /// `codes` joined at bit `LOW` with the row's `bits`-bit codes (1 or 2)
    /// of an `mma16` high plane: each at most 3, so no shift crosses its
    /// byte.
    #[inline(always)]
    pub(super) unsafe fn with_high<const LOW: u32>(codes: __m256i, lanes: Lanes, bits: u32) -> __m256i {
        unsafe {
            let high = _mm256_sll_epi16(lane_bits(lanes, bits), _mm_cvtsi32_si128(LOW as i32));
            _mm256_or_si256(codes, high)
        }
    }

    /// The row's 8-bit codes of an `mma16` codes plane in element order.
    #[inline(always)]
    pub(super) unsafe fn lane_bytes(lanes: Lanes) -> __m256i {
        unsafe {
            let select = _mm256_add_epi8(constant(&SELECT), _mm256_set1_epi8(lanes.upper as i8));
            let (first, second) = (
                _mm256_loadu2_m128i(lanes.lane(1).cast(), lanes.lane(0).cast()),
                _mm256_loadu2_m128i(lanes.lane(3).cast(), lanes.lane(2).cast()),
            );
            let (first, second) = (
                _mm256_shuffle_epi8(first, select),
                _mm256_shuffle_epi8(second, select),
            );
            steps(
                _mm256_unpacklo_epi64(first, second),
                _mm256_setr_epi32(0, 4, 2, 6, 1, 5, 3, 7),
            )
        }
    }
}

/// Integer dot of 32 signed weight values and signed activation codes.
#[inline(always)]
fn dot_i8_i8<V: Vectors>(weights: &[i8; 32], activations: &[i8; 32]) -> i32 {
    #[cfg(target_arch = "x86_64")]
    if V::QUANTIZED_DOTS >= 1 {
        return unsafe {
            let (w, a) = (
                packed_x86::load(weights.as_ptr()),
                packed_x86::load(activations.as_ptr()),
            );
            packed_x86::sum(packed_x86::dot_signed::<V>(
                std::arch::x86_64::_mm256_setzero_si256(),
                w,
                a,
            ))
        };
    }
    #[cfg(target_arch = "aarch64")]
    if V::QUANTIZED_DOTS >= 1 {
        return unsafe { packed_neon::dot32(weights.as_ptr(), activations.as_ptr()) };
    }
    #[allow(unreachable_code)]
    weights
        .iter()
        .zip(activations)
        .map(|(w, a)| i32::from(*w) * i32::from(*a))
        .sum()
}

#[inline(always)]
unsafe fn read<T: Copy>(pointer: *const u8) -> T {
    unsafe { pointer.cast::<T>().read_unaligned() }
}

/// The body of the row-decode component: the `k` values of one row.
///
/// # Safety
/// `row` addresses a row of `W` with `geometry` and `k` values; `out` holds
/// at least `k` values.
#[inline(always)]
pub unsafe fn decode_row<W: Format, V: Vectors>(
    row: *const u8,
    geometry: &RowGeometry,
    k: usize,
    out: &mut [f32],
) {
    let out = &mut out[..k];
    let row = geometry.row(row);
    let mut storage = V::ZERO;
    let packet = V::packet(&mut storage);
    for (p, chunk) in out.chunks_mut(PACKET).enumerate() {
        unsafe { W::decode::<V>(&row, p, k, packet) };
        chunk.copy_from_slice(&packet[..chunk.len()]);
    }
}

/// `lanes` plus the products of a packet's values and activations, value
/// `i` fused into lane `i % LANES` in index order: the tier's vector
/// multiply-adds where it has them, so the accumulation is the same
/// instructions in every component.
#[inline(always)]
unsafe fn accumulate<V: Vectors>(
    lanes: &mut [f32; LANES],
    values: &[f32; PACKET],
    x: &[f32; PACKET],
) {
    const { assert!(LANES == 8 && PACKET % LANES == 0, "a lane vector of eight") };
    #[cfg(target_arch = "x86_64")]
    if V::QUANTIZED_DOTS >= 1 {
        use std::arch::x86_64::*;
        unsafe {
            let mut sum = _mm256_loadu_ps(lanes.as_ptr());
            for chunk in 0..PACKET / LANES {
                let at = LANES * chunk;
                sum = _mm256_fmadd_ps(
                    _mm256_loadu_ps(values.as_ptr().add(at)),
                    _mm256_loadu_ps(x.as_ptr().add(at)),
                    sum,
                );
            }
            _mm256_storeu_ps(lanes.as_mut_ptr(), sum);
        }
        return;
    }
    #[cfg(target_arch = "aarch64")]
    unsafe {
        use std::arch::aarch64::*;
        let (mut low, mut high) = (vld1q_f32(lanes.as_ptr()), vld1q_f32(lanes.as_ptr().add(4)));
        for chunk in 0..PACKET / LANES {
            let at = LANES * chunk;
            low = vfmaq_f32(
                low,
                vld1q_f32(values.as_ptr().add(at)),
                vld1q_f32(x.as_ptr().add(at)),
            );
            high = vfmaq_f32(
                high,
                vld1q_f32(values.as_ptr().add(at + 4)),
                vld1q_f32(x.as_ptr().add(at + 4)),
            );
        }
        vst1q_f32(lanes.as_mut_ptr(), low);
        vst1q_f32(lanes.as_mut_ptr().add(4), high);
    }
    #[cfg(not(target_arch = "aarch64"))]
    for chunk in 0..PACKET / LANES {
        for lane in 0..LANES {
            let i = chunk * LANES + lane;
            lanes[lane] = values[i].mul_add(x[i], lanes[lane]);
        }
    }
}

/// The body of the dot component: rows `0..R` (from `rows`, `geometry.stride`
/// apart) against the `k` values of `x`.
///
/// # Safety
/// `rows` addresses `R` rows of `W` with `geometry` and `k` values.
#[inline(always)]
pub unsafe fn dot_rows<W: Format, const R: usize, V: Vectors>(
    rows: *const u8,
    geometry: &RowGeometry,
    x: &[f32],
    k: usize,
) -> [f32; R] {
    let x = &x[..k];
    let mut lanes = [[0.0f32; LANES]; R];
    // One packet buffer, consumed by its row before the next row decodes:
    // with every row's packet live at once, x86 vectorizers transpose the
    // accumulation across rows into a gather per multiply-add.
    let mut storage = V::ZERO;
    let packet = V::packet(&mut storage);
    let rows =
        geometry.rows::<R>(rows);
    let full = k / PACKET;
    for p in 0..full {
        let xs: &[f32; PACKET] = x[p * PACKET..(p + 1) * PACKET]
            .try_into()
            .expect("a whole packet");
        for (r, lanes) in lanes.iter_mut().enumerate() {
            unsafe { W::dot_packet::<V>(&rows[r], p, k, lanes, xs, packet) };
        }
    }
    let tail = k - full * PACKET;
    if tail > 0 {
        let xs = &x[full * PACKET..];
        for (r, lanes) in lanes.iter_mut().enumerate() {
            unsafe { W::decode::<V>(&rows[r], full, k, packet) };
            for (i, value) in xs.iter().enumerate() {
                lanes[i % LANES] = packet[i].mul_add(*value, lanes[i % LANES]);
            }
        }
    }
    lanes.map(combine)
}

/// `dot_q8_rows`'s sums for `rows8` rows of an element with the tile form,
/// in `out` (`[a n + r]` for the `n` rows and `m` activation rows), each
/// activation row a strip of one: every row block of the span in one call.
///
/// # Safety
/// As [`dot_q8_rows`], with `out` holding `m * n` sums and the rows
/// `rows8` rows of `W`, which has the tile form (`Format::TILED`).
#[inline(always)]
pub unsafe fn dot_q8_tile<W: Format, V: Vectors>(
    rows: *const u8,
    geometry: &RowGeometry,
    n: usize,
    x: &[Q8Block],
    m: usize,
    k: usize,
    out: &mut [f32],
) {
    let x = Q8Block::as_tiles(&x[..m * crate::quant::blocks(k)]);
    unsafe { W::tile_q8::<V, 1>(rows, geometry, n, x, m, k, out) };
}

/// The body of the quantized dot component: rows `0..R` (from `rows`,
/// `geometry.stride` apart) of `k` values against the `M` activation rows
/// quantized into `x` (`quant::blocks(k)` blocks each, row after row),
/// activation-major: `[m][r]` accumulates row `r`'s block dots with
/// activation row `m` in block order. One pass over the weight rows decodes
/// every block's codes and coefficients once for all `M`
/// ([`Format::dot_q8`]), whose sums are the same for every `M`: so every
/// activation row's sums are the ones it has alone.
///
/// # Safety
/// `rows` addresses `R` rows of `W` with `geometry` and `k` values, and `x`
/// holds `quant::blocks(k)` blocks of each of `M` activation rows.
#[inline(always)]
pub unsafe fn dot_q8_rows<W: Format, const R: usize, const M: usize, V: Vectors>(
    rows: *const u8,
    geometry: &RowGeometry,
    x: &[Q8Block],
    k: usize,
) -> [[f32; R]; M] {
    let blocks = crate::quant::blocks(k);
    let mut sums = [[0.0f32; R]; M];
    // Loops, not `array::from_fn`: a closure is a type of its own, so each
    // would instantiate the std helpers once for every format, tier and shape.
    const { assert!(M >= 1, "at least one activation row") };
    let mut x_rows: [&[Q8Block]; M] = [&x[..blocks]; M];
    for (m, row) in x_rows.iter_mut().enumerate() {
        *row = &x[m * blocks..(m + 1) * blocks];
    }
    let x = x_rows;
    if R == 16 && M == 1 && unsafe { W::dot_q8_tiles::<V>(rows, geometry, x[0], k, &mut sums[0]) } {
        return sums;
    }
    let rows = geometry.rows::<R>(rows);
    for b in 0..blocks {
        let mut block = [&x[0][b]; M];
        for (m, block) in block.iter_mut().enumerate() {
            *block = &x[m][b];
        }
        let values = unsafe { W::dot_q8::<R, M, V>(&rows, b, k, block) };
        for (r, values) in values.iter().enumerate() {
            for (sums, value) in sums.iter_mut().zip(values) {
                sums[r] += value;
            }
        }
    }
    sums
}

#[cfg(test)]
mod tests {
    use crate::components::{elements, resolve};
    use crate::isa::Tier;
    use seismic_lang::registry::{self, CodingAxes, Layout, RepresentationKind, RowPlaneRole};

    /// Every packed weight element decodes rows of random codes to exactly
    /// the registry's decode of the same bytes.
    #[test]
    fn packed_decode_matches_the_registry() {
        use seismic_lang::interp::TensorData;
        let tier = Tier::detected().unwrap();
        // Each element's first mismatch.
        let mut mismatches = Vec::new();
        'element: for name in elements() {
            let id = registry::representation(name).unwrap();
            let info = registry::representation_info(id);
            let RepresentationKind::PackedRows(layout) = &info.kind else {
                continue;
            };
            if info.layout != Layout::Rows16 {
                continue;
            }
            let kernels = resolve(tier, id).unwrap();
            let axes = registry::decode_axes(id).unwrap();
            let encoding = match axes.coding {
                CodingAxes::Direct { scale, .. } => scale,
                CodingAxes::Hierarchical { factor, .. } => factor,
            };
            let group = layout.group() as usize;
            // Also the Gemma K = 704 and the Nemotron K = 2688 where whole
            // storage groups make them.
            let lengths = [1, 3, 10].map(|groups| groups * group);
            for k in lengths
                .into_iter()
                .chain([704, 2688].into_iter().filter(|k| k % group == 0))
            {
                let geometry = super::RowGeometry::of(info, k, 0);
                let rows = 3;
                let mut state = 0x2545_f491_4f6c_dd1du64 ^ k as u64;
                let mut data = (0..rows * geometry.stride)
                    .map(|_| {
                        state ^= state << 13;
                        state ^= state >> 7;
                        state ^= state << 17;
                        (state >> 24) as u8
                    })
                    .collect::<Vec<_>>();
                // Finite coefficients: moderate floats (random f16 and f32
                // bytes would be NaN or overflow), and every finite code of
                // the scale codes.
                let supers = RowPlaneRole::Supers as usize;
                let bytes = layout.row_groups(k as u64).unwrap() as usize * geometry.groups[supers];
                for row in 0..rows {
                    let start = row * geometry.stride + geometry.offsets[supers];
                    let plane = &mut data[start..start + bytes];
                    crate::element::finite_scales(plane, encoding, true);
                }
                let reference = TensorData::encoded(id, vec![rows, k], data.clone())
                    .unwrap()
                    .values()
                    .unwrap();
                let mut out = vec![0.0f32; k];
                for row in 0..rows {
                    unsafe {
                        kernels.decode(
                            data.as_ptr().add(row * geometry.stride),
                            &geometry,
                            k,
                            &mut out,
                        )
                    };
                    // Equal values, a NaN bit for bit the registry's (its
                    // one canonical NaN).
                    let mismatch = out.iter().enumerate().find_map(|(i, value)| {
                        let expected = reference[row * k + i] as f32;
                        let same = if expected.is_nan() {
                            value.to_bits() == expected.to_bits()
                        } else {
                            *value == expected
                        };
                        (!same).then(|| {
                            format!(
                                "{name} k {k} row {row} value {i}: {:#010x} is not {:#010x}",
                                value.to_bits(),
                                expected.to_bits()
                            )
                        })
                    });
                    if let Some(mismatch) = mismatch {
                        mismatches.push(mismatch);
                        continue 'element;
                    }
                }
            }
        }
        assert!(mismatches.is_empty(), "{}", mismatches.join("\n"));
    }

    /// Every element a row conversion writes, in its own layout (the CPU's
    /// and the devices' alike, tiles included), decodes through the CPU's
    /// components to exactly the registry's decode of the converted bytes:
    /// what a host reference that decodes such records natively (a CPU
    /// site's admission) stands on.
    #[test]
    fn every_row_conversion_decodes_as_the_registry() {
        use seismic_lang::interp::TensorData;
        use seismic_lang::registry::ConversionLayout;
        let tier = Tier::detected().unwrap();
        let mut checked = 0;
        for conversion in registry::representation_conversions() {
            let ConversionLayout::Rows(layout) = conversion.destination_layout else {
                continue;
            };
            let Some(operand) = crate::WeightOperand::of(tier, conversion.destination) else {
                continue;
            };
            let name = registry::representation_info(conversion.destination).name;
            // Whole tiles, and whole storage groups and source packets of
            // 32 and 256 values.
            let (rows, k) = (2 * layout.tile_rows() as usize, 768usize);
            assert_eq!(k % layout.group() as usize, 0, "{name}");
            let extents = [rows as u64, k as u64];
            let length = registry::canonical_bytes(conversion.source, &extents)
                .unwrap_or_else(|| panic!("{name}: no source bytes of {extents:?}"));
            let mut state = 0x2545_f491_4f6c_dd1du64 ^ conversion.id.position() as u64;
            let mut source = (0..length)
                .map(|_| {
                    state ^= state << 13;
                    state ^= state >> 7;
                    state ^= state << 17;
                    (state >> 24) as u8
                })
                .collect::<Vec<_>>();
            conversion.finite_source(&mut source);
            let converted = crate::repack::convert(conversion, &extents, &source).unwrap();
            let reference = TensorData::encoded(conversion.destination, vec![rows, k], converted.clone())
                .unwrap()
                .values()
                .unwrap();
            // SAFETY: `converted` is the destination's canonical bytes of
            // `extents`.
            let weights = unsafe {
                crate::Weights::from_tensor(converted.as_ptr(), &extents, &[k as u64, 1], operand)
            };
            let mut out = vec![0.0f32; k];
            for row in 0..rows {
                weights.decode_row(row, &mut out);
                for (i, value) in out.iter().enumerate() {
                    assert_eq!(f64::from(*value), reference[row * k + i], "{name} row {row} value {i}");
                }
            }
            checked += 1;
        }
        assert!(checked > 0, "no row conversion's element is read by the CPU");
    }
}
