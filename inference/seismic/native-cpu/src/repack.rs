//! Exact conversions of external GGUF weights into resident storage: the
//! registry's registered conversions, executed from their recipes'
//! executable form ([`program`]). [`convert`] converts a whole tensor into
//! any resident layout; into resident `rows16` and `rows8` storage, one
//! source packet becomes one storage group of its row.
//!
//! A kernel converts through [`External`] and [`Packed`] views of its bound
//! tensors; [`rows`] resolves the conversion from their representations.

mod program;

use program::{PacketForms, RepackExecutor, RepackProgram, BLOCK, EXECUTION_BYTES};
pub use program::{ScaledCodeRows, ScaledCodesPlacement};
use seismic_lang::ids::RepresentationId;
use seismic_lang::registry::{
    self, ConversionLayout, Layout, PackedRowLayout, RepresentationConversion, RepresentationInfo,
    RepresentationKind, RowGeometry,
};
use std::marker::PhantomData;
use std::sync::OnceLock;

/// Values of each row of a tile one execution of `RowConversion::chunks`
/// converts at least, and so the most a row converted whole has: the rows
/// of a conversion are converted in such chunks only when they are longer.
/// Measured (`repack_throughput`, M4 Max): sizing a chunk by
/// [`EXECUTION_BYTES`] across the tile's rows instead chunks `rows8` rows of
/// 4096 values, which then convert up to 2.6 times slower (`q5_k`, 11.2 to
/// 4.3 GB/s on one thread), and changes no `rows16` conversion beyond noise.
const CHUNK_VALUES: usize = 4096;

/// Tiles of `tile` rows of `row_bytes` source bytes an execution of whole
/// rows converts at most: the fewest, a power of two, reading at least
/// [`EXECUTION_BYTES`] bytes.
fn whole_tiles(tile: usize, row_bytes: usize) -> usize {
    EXECUTION_BYTES
        .div_ceil(tile * row_bytes)
        .next_power_of_two()
}

/// One registered conversion into a resident row layout as a kernel
/// resolves it.
#[derive(Debug)]
pub struct RowConversion {
    conversion: &'static RepresentationConversion,
    layout: &'static PackedRowLayout,
    /// Rows of a tile of the layout.
    tile: usize,
    /// Source packets of a row one execution of [`chunks`](Self::chunks)
    /// converts.
    packets: usize,
    /// The program of [`chunks`](Self::chunks), compiled when first
    /// converting rows of more than `packets` packets.
    chunks: OnceLock<RepackProgram>,
    /// The programs of whole rows of at most `packets` packets, by row
    /// length (`packets - 1`) and log2 of tiles (up to [`whole_tiles`] of
    /// the length), each compiled when first converting such rows
    /// ([`whole_rows`](Self::whole_rows)).
    whole: Box<[WholeRows]>,
}

/// The programs of whole rows of one length, by log2 of tiles.
type WholeRows = Box<[OnceLock<Box<RepackProgram>>]>;

impl RowConversion {
    fn of(conversion: &'static RepresentationConversion) -> Option<Self> {
        let ConversionLayout::Rows(layout) = conversion.destination_layout else {
            return None;
        };
        if !matches!(layout.layout, Layout::Rows16 | Layout::Rows8) {
            return None;
        }
        let external = conversion.source_layout;
        let (group, packet_bytes) = (
            external.logical_group as usize,
            external.packet_size as usize,
        );
        let tile = layout.tile_rows() as u32;
        // At least `CHUNK_VALUES` values of each row, and whole blocks of
        // each plane.
        let least = CHUNK_VALUES.div_ceil(group) as u32;
        let packets = (1..)
            .map(|times| times * least)
            .find(|packets| {
                layout.planes.iter().all(|plane| {
                    ((packets * tile * plane.bytes_per_group) as usize).is_multiple_of(BLOCK)
                })
            })
            .expect("some multiple of an execution is whole blocks") as usize;
        let tile = tile as usize;
        Some(Self {
            conversion,
            layout,
            tile,
            packets,
            chunks: OnceLock::new(),
            whole: (1..=packets)
                .map(|packets| {
                    (0..=whole_tiles(tile, packets * packet_bytes).ilog2())
                        .map(|_| OnceLock::new())
                        .collect()
                })
                .collect(),
        })
    }

    /// The source's representation.
    pub fn source(&self) -> &'static str {
        registry::representation_info(self.conversion.source).name
    }

    /// The target's representation.
    pub fn target(&self) -> &'static str {
        registry::representation_info(self.conversion.destination).name
    }

    /// Values of one source packet.
    pub fn group(&self) -> usize {
        self.conversion.source_layout.logical_group as usize
    }

    /// Bytes of one source packet.
    fn packet_bytes(&self) -> usize {
        self.conversion.source_layout.packet_size as usize
    }

    /// The recipe over `packets` packets of each row of a tile, each row an
    /// input segment, into a segment per plane of the target's layout (in
    /// plane order) holding those groups of the tile's rows as the layout
    /// places them: rows longer than `packets` packets convert in such
    /// chunks.
    fn chunks(&self) -> &RepackProgram {
        self.chunks.get_or_init(|| self.compile_chunks())
    }

    fn compile_chunks(&self) -> RepackProgram {
        let (layout, tile, packets) = (self.layout, self.tile as u32, self.packets as u32);
        let placed = layout
            .geometry(u64::from(packets) * self.group() as u64)
            .expect("an execution's row geometry");
        // Plane `p` of a placed tile lies at `offsets[p] * tile`.
        let planes = placed
            .offsets
            .iter()
            .zip(&placed.bytes_per_row)
            .map(|(offset, bytes)| (offset * u64::from(tile), bytes * u64::from(tile)))
            .collect::<Vec<_>>();
        RepackProgram::compile(
            self.conversion,
            tile,
            packets,
            &planes
                .iter()
                .map(|(_, bytes)| *bytes as u32)
                .collect::<Vec<_>>(),
            |row, packet, plane, bit| {
                if !layout.occupies(plane, bit) {
                    return None;
                }
                let at = layout.packet_bit(&placed, u64::from(row), u64::from(packet), plane, bit);
                let index = planes
                    .iter()
                    .position(|(offset, bytes)| (offset * 8..(offset + bytes) * 8).contains(&at))
                    .expect("a placed bit lies in a plane");
                Some((index, (at - planes[index].0 * 8) as u32))
            },
        )
        .expect("a tile of an execution's chunk of a row fits a program")
    }

    /// Tiles an execution of whole rows of `packets` packets converts at
    /// most ([`whole_tiles`]).
    fn whole_tiles(&self, packets: usize) -> usize {
        whole_tiles(self.tile, packets * self.packet_bytes())
    }

    /// The program of `tiles` tiles (a power of two) of whole rows of
    /// `packets` packets (at most an execution's of `chunks`): the recipe
    /// over the rows' packets, one input segment, into their stored tiles,
    /// one output segment, every byte no plane occupies zero.
    fn whole_rows(&self, packets: usize, tiles: usize) -> &RepackProgram {
        self.whole[packets - 1][tiles.ilog2() as usize]
            .get_or_init(|| Box::new(self.compile_whole_rows(packets, tiles)))
    }

    fn compile_whole_rows(&self, packets: usize, tiles: usize) -> RepackProgram {
        let layout = self.layout;
        let rows = tiles * self.tile;
        let geometry = layout
            .geometry((packets * self.group()) as u64)
            .expect("a row's geometry");
        // Tile `t` of the execution lies `t` tiles of stored rows past the
        // first; within it, the layout places the tile's rows' packets.
        let tile_bits = (self.tile as u64 * geometry.stride * 8) as u32;
        RepackProgram::compile(
            self.conversion,
            1,
            (rows * packets) as u32,
            &[(rows as u64 * geometry.stride) as u32],
            |_, packet, plane, bit| {
                if !layout.occupies(plane, bit) {
                    return None;
                }
                let (row, packet) = (packet as usize / packets, packet as usize % packets);
                let at = layout.packet_bit(
                    &geometry,
                    (row % self.tile) as u64,
                    packet as u64,
                    plane,
                    bit,
                );
                Some((0, (row / self.tile) as u32 * tile_bits + at as u32))
            },
        )
        .expect("an execution's whole rows fit a program")
    }
}

/// A registered conversion's CPU executors.
struct Executors {
    /// The recipe's program writing packet forms, compiled when first
    /// converting a whole tensor ([`convert`]).
    forms: OnceLock<PacketForms>,
    /// Its form when it converts into a resident row layout.
    rows: Option<RowConversion>,
}

/// Every registered conversion's (by id) CPU executors.
fn executors() -> &'static [Executors] {
    static EXECUTORS: OnceLock<Vec<Executors>> = OnceLock::new();
    EXECUTORS.get_or_init(|| {
        registry::representation_conversions()
            .iter()
            .map(|conversion| Executors {
                forms: OnceLock::new(),
                rows: RowConversion::of(conversion),
            })
            .collect()
    })
}

/// The canonical destination bytes of `conversion` over `source`, the
/// canonical source bytes of a tensor of `extents`: every packet through
/// the recipe's program (across threads when there are many), then placed
/// by the destination layout. `None` when `source` is not the source's
/// canonical byte count over `extents`.
pub fn convert(
    conversion: &RepresentationConversion,
    extents: &[u64],
    source: &[u8],
) -> Option<Vec<u8>> {
    if registry::canonical_bytes(conversion.source, extents)? != source.len() as u64 {
        return None;
    }
    let forms = executors()[conversion.id.index()]
        .forms
        .get_or_init(|| PacketForms::compile(conversion))
        .execute(source);
    Some(match conversion.destination_layout {
        ConversionLayout::Rows(layout) => layout.place(extents, &forms),
        ConversionLayout::Packet(_) => forms,
    })
}

/// Every registered conversion into a resident row layout.
pub fn conversions() -> impl Iterator<Item = &'static RowConversion> {
    executors()
        .iter()
        .filter_map(|conversion| conversion.rows.as_ref())
}

/// The conversion of `source` into `target`, when one is registered.
pub fn resolve(source: &str, target: &str) -> Option<&'static RowConversion> {
    conversion(
        registry::representation(source)?,
        registry::representation(target)?,
    )
}

/// The conversion of `source` into `target`, when one is registered.
fn conversion(
    source: RepresentationId,
    target: RepresentationId,
) -> Option<&'static RowConversion> {
    executors()[registry::representation_conversion(source, target)?
        .id
        .index()]
    .rows
    .as_ref()
}

/// Leading-axis row addressing shared by [`External`] and [`Packed`]: a
/// tensor's rows are the positions of every axis before the last, which a
/// canonical view stores `row_stride` apart, each matrix's rows padded to
/// whole tiles of `tile` rows.
fn canonical_rows(extents: &[u64], strides: &[u64], row_stride: u64, tile: u64) -> usize {
    let rank = extents.len();
    assert!(rank >= 2, "a converted operand has a row and a value axis");
    let mut expected = row_stride;
    for axis in (0..rank - 1).rev() {
        assert_eq!(
            strides[axis], expected,
            "a converted operand's rows are canonical"
        );
        expected *= if axis == rank - 2 {
            extents[axis].next_multiple_of(tile)
        } else {
            extents[axis]
        };
    }
    extents[..rank - 1].iter().product::<u64>() as usize
}

/// A bound external source tensor: whole packets per row, read only by a
/// conversion.
#[derive(Clone, Copy, Debug)]
pub struct External<'a> {
    base: *const u8,
    rows: usize,
    k: usize,
    id: RepresentationId,
    matrix_rows: usize,
    life: PhantomData<&'a u8>,
}

unsafe impl Send for External<'_> {}
unsafe impl Sync for External<'_> {}

impl<'a> External<'a> {
    /// The external tensor at `base` with `extents` in values and `strides`
    /// in packets.
    ///
    /// # Safety
    /// `base`, `extents` and `strides` describe a tensor of `representation`
    /// valid for `'a`.
    pub unsafe fn from_tensor(
        base: *const u8,
        extents: &[u64],
        strides: &[u64],
        representation: &'static str,
    ) -> Self {
        let k = extents[extents.len() - 1];
        let id = registry::representation(representation)
            .unwrap_or_else(|| panic!("`{representation}` is no registered representation"));
        let (group, _) = external_packet(registry::representation_info(id));
        let rows = canonical_rows(extents, strides, k.div_ceil(group as u64), 1);
        Self {
            base,
            rows,
            k: k as usize,
            id,
            matrix_rows: extents[extents.len() - 2] as usize,
            life: PhantomData,
        }
    }
}

/// The values and bytes of a packet of the external encoding `element`.
fn external_packet(element: &RepresentationInfo) -> (usize, usize) {
    let RepresentationKind::External(layout) = &element.kind else {
        panic!("`{}` is no external source", element.name)
    };
    (layout.logical_group as usize, layout.packet_size as usize)
}

/// A bound result tensor of a packed `rows16` or `rows8` element, written
/// only by a conversion.
#[derive(Clone, Copy, Debug)]
pub struct Packed<'a> {
    base: *mut u8,
    rows: usize,
    k: usize,
    element: &'static RepresentationInfo,
    matrix_rows: usize,
    life: PhantomData<&'a u8>,
}

unsafe impl Send for Packed<'_> {}
unsafe impl Sync for Packed<'_> {}

impl<'a> Packed<'a> {
    /// The packed tensor at `base` with `extents` in values and `strides` in
    /// rows.
    ///
    /// # Safety
    /// `base`, `extents` and `strides` describe a tensor of `representation`
    /// valid for `'a`.
    pub unsafe fn from_tensor(
        base: *mut u8,
        extents: &[u64],
        strides: &[u64],
        representation: &'static str,
    ) -> Self {
        let element = registry::representation_info(
            registry::representation(representation)
                .unwrap_or_else(|| panic!("`{representation}` is no registered element")),
        );
        let RepresentationKind::PackedRows(layout) = &element.kind else {
            panic!("`{representation}` is no row-layout element")
        };
        let rows = canonical_rows(extents, strides, 1, layout.tile_rows());
        Self {
            base,
            rows,
            k: extents[extents.len() - 1] as usize,
            element,
            matrix_rows: extents[extents.len() - 2] as usize,
            life: PhantomData,
        }
    }
}

/// Converts rows `rows` of `source` into the same rows of `destination`,
/// zeroing every stored byte no plane occupies.
///
/// # Safety
/// No other work item writes these rows of `destination` in the launch.
pub unsafe fn rows(source: &External<'_>, destination: &Packed<'_>, rows: std::ops::Range<usize>) {
    assert_eq!(source.k, destination.k, "a conversion keeps the row length");
    assert!(
        rows.end <= source.rows && rows.end <= destination.rows,
        "rows {rows:?} outside the operands"
    );
    assert_eq!(source.matrix_rows, destination.matrix_rows);
    PLANS.with_borrow_mut(|plans| {
        let key = (source.id, destination.element.id, source.k);
        let index = plans
            .iter()
            .position(|plan| plan.key == key)
            .unwrap_or_else(|| {
                plans.push(Plan::of(source, destination));
                plans.len() - 1
            });
        let Plan {
            rows: facts, form, ..
        } = &mut plans[index];
        let (tile, n) = (facts.conversion.tile, source.matrix_rows);
        if tile > 1 {
            assert!(
                rows.start / n == (rows.end - 1) / n,
                "a tiled work item stays within one matrix"
            );
            assert_eq!(
                rows.start % n % tile,
                0,
                "a tiled work item starts on a tile"
            );
            assert!(
                rows.end.is_multiple_of(n) || (rows.end % n).is_multiple_of(tile),
                "a tiled work item ends on a tile"
            );
        }
        // SAFETY: as this function's.
        unsafe {
            match form {
                Form::Whole(executions) => facts.whole(executions, source, destination, rows),
                Form::Chunks(chunks) => facts.chunks(chunks, source, destination, rows),
            }
        }
    });
}

/// How one thread converts rows of one length from one source into one
/// target: what [`rows`] derives once, and the executors of its programs.
struct Plan {
    /// The source's and the target's representations, and the row length.
    key: (RepresentationId, RepresentationId, usize),
    rows: Rows,
    form: Form,
}

/// The rows of a plan.
struct Rows {
    conversion: &'static RowConversion,
    geometry: RowGeometry,
    /// Packets and bytes of a source row.
    row_packets: usize,
    source_row: usize,
}

/// How a plan's rows convert.
enum Form {
    /// Whole rows by the programs of their length, of each power of two of
    /// tiles up to an execution's (`RowConversion::whole_tiles`): the
    /// executor of each, made when first run.
    Whole(Vec<Option<RepackExecutor<'static>>>),
    /// Chunks of each row of a tile by `RowConversion::chunks`.
    Chunks(Box<Chunks>),
}

/// The bytes of a tile no plane of a chunked plan occupies; the executor of
/// `RowConversion::chunks`, and the bases of an execution in place: per row
/// of a tile its source, per plane its output.
struct Chunks {
    gaps: Vec<(usize, usize)>,
    executor: RepackExecutor<'static>,
    sources: Vec<*const u8>,
    outputs: Vec<*mut u8>,
}

impl Plan {
    fn of(source: &External<'_>, destination: &Packed<'_>) -> Self {
        let conversion = conversion(source.id, destination.element.id).unwrap_or_else(|| {
            panic!(
                "no CPU conversion of `{}` into `{}`",
                registry::representation_info(source.id).name,
                destination.element.name
            )
        });
        let row_packets = source.k.div_ceil(conversion.group());
        let rows = Rows {
            conversion,
            geometry: conversion
                .layout
                .geometry(source.k as u64)
                .expect("a row's geometry"),
            row_packets,
            source_row: row_packets * conversion.packet_bytes(),
        };
        let form = if row_packets <= conversion.packets {
            let tiles = conversion.whole_tiles(row_packets);
            Form::Whole((0..=tiles.ilog2()).map(|_| None).collect())
        } else {
            let mut occupied = (0..conversion.layout.planes.len())
                .map(|plane| rows.shares(plane, 0..conversion.tile, 0..row_packets))
                .collect::<Vec<_>>();
            occupied.sort_by_key(|range| range.start);
            let (mut gaps, mut end) = (Vec::new(), 0);
            for range in occupied {
                if range.start > end {
                    gaps.push((end, range.start));
                }
                end = end.max(range.end);
            }
            let stored = conversion.tile * rows.stride();
            if stored > end {
                gaps.push((end, stored));
            }
            Form::Chunks(Box::new(Chunks {
                gaps,
                executor: conversion.chunks().executor(),
                sources: vec![std::ptr::null(); conversion.tile],
                outputs: vec![std::ptr::null_mut(); conversion.layout.planes.len()],
            }))
        };
        Self {
            key: (source.id, destination.element.id, source.k),
            rows,
            form,
        }
    }
}

impl Rows {
    /// Bytes between the starts of adjacent stored rows.
    fn stride(&self) -> usize {
        self.geometry.stride as usize
    }

    /// The bytes of a tile, from its first, that the shares of storage
    /// groups `groups` of rows `rows` of the tile occupy in plane `plane`:
    /// those of the first row's first group to those of the last row's
    /// last, which the layout places one after another
    /// (`PackedRowLayout::share_byte`).
    fn shares(
        &self,
        plane: usize,
        rows: std::ops::Range<usize>,
        groups: std::ops::Range<usize>,
    ) -> std::ops::Range<usize> {
        let layout = self.conversion.layout;
        let share = |row: usize, group: usize| {
            layout
                .share_byte(&self.geometry, row as u64, group as u64, plane)
                .expect("rows16 and rows8 keep a share's bytes together") as usize
        };
        share(rows.start, groups.start)
            ..share(rows.end - 1, groups.end - 1) + layout.planes[plane].bytes_per_group as usize
    }

    /// Zeroes the shares of storage groups `groups` of rows `rows` of the
    /// tile `tile` in every plane.
    fn zero(&self, tile: &mut [u8], rows: std::ops::Range<usize>, groups: std::ops::Range<usize>) {
        for plane in 0..self.conversion.layout.planes.len() {
            for row in rows.clone() {
                for group in groups.clone() {
                    tile[self.shares(plane, row..row + 1, group..group + 1)].fill(0);
                }
            }
        }
    }

    /// The first stored byte of the tile of `row`: in `rows8` padded to
    /// whole tiles per matrix.
    fn stored(&self, destination: &Packed<'_>, row: usize) -> *mut u8 {
        let (tile, n) = (self.conversion.tile, destination.matrix_rows);
        let stored = row / n * n.div_ceil(tile) * tile + row % n;
        // SAFETY: the row lies in the destination (`from_tensor`).
        unsafe { destination.base.add(stored * self.stride()) }
    }

    /// Converts `rows` by executions of whole tiles: as many as `executions`
    /// has programs of, the last the fewest a power of two of them covering
    /// the rows left.
    ///
    /// # Safety
    /// As [`rows`]'s.
    unsafe fn whole(
        &self,
        executions: &mut [Option<RepackExecutor<'static>>],
        source: &External<'_>,
        destination: &Packed<'_>,
        rows: std::ops::Range<usize>,
    ) {
        let (conversion, stride, source_row) = (self.conversion, self.stride(), self.source_row);
        let tile = conversion.tile;
        let source_bytes = source.rows * source_row;
        let most = tile << (executions.len() - 1);
        for first in rows.clone().step_by(most) {
            let present = (rows.end - first).min(most);
            let tiles = present.div_ceil(tile).next_power_of_two();
            let executor = executions[tiles.ilog2() as usize]
                .get_or_insert_with(|| conversion.whole_rows(self.row_packets, tiles).executor());
            // SAFETY: (all below) this work item owns the rows, and in
            // `rows8` their entire tiles; the stored tiles of an execution's
            // rows follow one another.
            let (input, output) = (
                unsafe { source.base.add(first * source_row) },
                self.stored(destination, first),
            );
            // In place when the execution's rows are all present and the
            // source holds a block past them.
            if present == tiles * tile && (first + present) * source_row + BLOCK <= source_bytes {
                unsafe { executor.run_into(&[input], &[output]) };
                continue;
            }
            // Past the rows present, the staging holds what an earlier
            // execution left: rows convert apart, and the padded rows of the
            // last tile are zeroed below.
            let bytes = present * source_row;
            executor.source()[..bytes]
                .copy_from_slice(unsafe { std::slice::from_raw_parts(input, bytes) });
            let placed = executor.run();
            let stored = present.div_ceil(tile);
            let out = unsafe { std::slice::from_raw_parts_mut(output, stored * tile * stride) };
            out.copy_from_slice(&placed[..out.len()]);
            // The rows the last tile pads with zero.
            let padded = stored * tile - present;
            self.zero(
                &mut out[(stored - 1) * tile * stride..],
                tile - padded..tile,
                0..self.row_packets,
            );
        }
    }

    /// Converts `rows` tile by tile, each row in chunks of
    /// `RowConversion::chunks`.
    ///
    /// # Safety
    /// As [`rows`]'s.
    unsafe fn chunks(
        &self,
        chunks: &mut Chunks,
        source: &External<'_>,
        destination: &Packed<'_>,
        rows: std::ops::Range<usize>,
    ) {
        let Chunks {
            gaps,
            executor,
            sources,
            outputs,
        } = chunks;
        let gaps = &*gaps;
        let conversion = self.conversion;
        // Where a staged execution holds each plane.
        let staged = conversion.chunks().output_starts();
        let (tile, row_packets, source_row) = (conversion.tile, self.row_packets, self.source_row);
        let source_bytes = source.rows * source_row;
        let execution = conversion.packets * conversion.packet_bytes();
        for first in rows.clone().step_by(tile) {
            let present = (rows.end - first).min(tile);
            // SAFETY: this work item owns the rows, and in `rows8` their
            // entire tile, including its padded rows.
            let out = unsafe {
                std::slice::from_raw_parts_mut(
                    self.stored(destination, first),
                    tile * self.stride(),
                )
            };
            for (start, end) in gaps {
                out[*start..*end].fill(0);
            }
            for (index, start) in (0..source_row).step_by(execution).enumerate() {
                let first_group = index * conversion.packets;
                let groups = first_group..row_packets.min(first_group + conversion.packets);
                // Each plane's shares of the groups of every row of the tile.
                let shares = |plane| self.shares(plane, 0..tile, groups.clone());
                // In place when every row of the tile has the execution's
                // packets and the source holds a block past them.
                if groups.len() == conversion.packets
                    && present == tile
                    && (first + tile - 1) * source_row + start + execution + BLOCK <= source_bytes
                {
                    for (r, base) in sources.iter_mut().enumerate() {
                        // SAFETY: the row's packets and a block past them lie
                        // in the source.
                        *base = unsafe { source.base.add((first + r) * source_row + start) };
                    }
                    // Every plane's base derives from one pointer to `out`,
                    // which nothing else references until the run is done.
                    let tile_base = out.as_mut_ptr();
                    for (plane, base) in outputs.iter_mut().enumerate() {
                        *base = tile_base.wrapping_add(shares(plane).start);
                    }
                    // SAFETY: as above; the planes' groups lie in `out`, each
                    // apart from the others.
                    unsafe { executor.run_into(sources, outputs) };
                    continue;
                }
                // Past each present row's groups, and in the rows the tile
                // pads, the staging holds what an earlier execution left:
                // groups convert apart, only the present groups are copied
                // out, and the padded rows are zeroed below.
                let bytes = groups.len() * conversion.packet_bytes();
                let input = executor.source();
                for (r, input) in input.chunks_exact_mut(execution).take(present).enumerate() {
                    // SAFETY: the row lies inside the source (`from_tensor`).
                    let row = unsafe {
                        std::slice::from_raw_parts(
                            source.base.add((first + r) * source_row + start),
                            bytes,
                        )
                    };
                    input[..bytes].copy_from_slice(row);
                }
                let placed = executor.run();
                // Each plane's groups of every row of the tile, then the rows
                // the tile pads with zero.
                for (plane, staged) in staged.iter().enumerate() {
                    let (to, staged) = (shares(plane), *staged as usize);
                    out[to.clone()].copy_from_slice(&placed[staged..staged + to.len()]);
                }
                self.zero(out, present..tile, groups);
            }
        }
    }
}

thread_local! {
    /// The plans of this thread, one per source, target and row length it
    /// has converted: a plan's derived facts and its executor's working
    /// memory are made once per thread, not per work item.
    static PLANS: std::cell::RefCell<Vec<Plan>> = const { std::cell::RefCell::new(Vec::new()) };
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::weights::RowGeometry;
    use seismic_lang::registry::{pseudo_random_bytes, representation};

    /// Every conversion reproduces the registry's host repack bit for bit,
    /// on rows whose last packet is partial, staged and in place: over
    /// tiles with rows missing and whole, executions of whole rows of every
    /// size with rows missing and whole, and rows of fewer and more packets
    /// than an execution's.
    #[test]
    fn conversions_match_the_registry_repack() {
        for conversion in conversions() {
            let execution = conversion.packets * conversion.group();
            for (rows, k) in [
                (3, conversion.group()),
                (3, 3 * conversion.group()),
                (3, 2560),
                (3, 2560 + 40),
                (17, 2560 + 40),
                (11, 2 * execution + conversion.group() + 40),
            ] {
                let packets = k.div_ceil(conversion.group());
                let mut source =
                    pseudo_random_bytes(k as u64, rows * packets * conversion.packet_bytes());
                let registered = seismic_lang::registry::representation_conversion(
                    representation(conversion.source()).unwrap(),
                    representation(conversion.target()).unwrap(),
                )
                .expect("registered conversion");
                // The scales a conversion converts as numbers are finite: NaN
                // payloads are not part of the conversion contract.
                registered.admit_source(&mut source);
                let id = registered.id;
                let expected =
                    seismic_lang::interp::repack(id, &[rows, k], &source).expect("registry repack");
                let geometry = RowGeometry::of(
                    registry::representation_info(representation(conversion.target()).unwrap()),
                    k,
                    0,
                );
                let stride = geometry.stride;
                let stored_rows = rows.next_multiple_of(conversion.tile);
                let mut out = vec![0xa5u8; stored_rows * stride];
                let from = unsafe {
                    External::from_tensor(
                        source.as_ptr(),
                        &[rows as u64, k as u64],
                        &[packets as u64, 1],
                        conversion.source(),
                    )
                };
                let to = unsafe {
                    Packed::from_tensor(
                        out.as_mut_ptr(),
                        &[rows as u64, k as u64],
                        &[1, 1],
                        conversion.target(),
                    )
                };
                // Work items of six tiles: some end on an execution of fewer
                // tiles than the most, in place or staged.
                let item = 6 * conversion.tile;
                for first in (0..rows).step_by(item) {
                    unsafe { super::rows(&from, &to, first..(first + item).min(rows)) };
                }
                assert_eq!(
                    out.len(),
                    expected.len(),
                    "{} -> {} k {k}",
                    conversion.source(),
                    conversion.target()
                );
                if let Some(at) = out.iter().zip(&expected).position(|(a, b)| a != b) {
                    let word = at / 4 * 4;
                    panic!(
                        "{} -> {} k {k}: byte {at} (row {}, offset {}) is {:#04x}, the registry's {:#04x}; words {:?} {:?}",
                        conversion.source(),
                        conversion.target(),
                        at / stride,
                        at % stride,
                        out[at],
                        expected[at],
                        f32::from_le_bytes(out[word..word + 4].try_into().unwrap()),
                        f32::from_le_bytes(expected[word..word + 4].try_into().unwrap()),
                    );
                }
            }
        }
    }

    /// Throughput of the CPU import over large random tensors of every GGUF
    /// source into each resident row layout, in GB/s of source bytes read:
    /// one thread, and every available thread over eight-row items.
    #[test]
    #[ignore = "measurement; run with --release --ignored --nocapture"]
    fn repack_throughput() {
        let threads = std::thread::available_parallelism().unwrap().get();
        let (rows, k) = (4096usize, 4096usize);
        for conversion in conversions() {
            let packets = k.div_ceil(conversion.group());
            let mut source = pseudo_random_bytes(0, rows * packets * conversion.packet_bytes());
            seismic_lang::registry::representation_conversion(
                representation(conversion.source()).unwrap(),
                representation(conversion.target()).unwrap(),
            )
            .unwrap()
            .admit_source(&mut source);
            let geometry = RowGeometry::of(
                registry::representation_info(representation(conversion.target()).unwrap()),
                k,
                0,
            );
            let mut out = vec![0u8; rows * geometry.stride];
            let from = unsafe {
                External::from_tensor(
                    source.as_ptr(),
                    &[rows as u64, k as u64],
                    &[packets as u64, 1],
                    conversion.source(),
                )
            };
            let to = unsafe {
                Packed::from_tensor(
                    out.as_mut_ptr(),
                    &[rows as u64, k as u64],
                    &[1, 1],
                    conversion.target(),
                )
            };
            // Each thread at the highest quality of service (on the
            // performance cores, where macOS schedules it), over the tensor
            // `PASSES` times so that spawning is a small part of the time.
            #[cfg(target_os = "macos")]
            fn interactive() {
                extern "C" {
                    fn pthread_set_qos_class_self_np(class: u32, priority: i32) -> i32;
                }
                // QOS_CLASS_USER_INTERACTIVE.
                assert_eq!(unsafe { pthread_set_qos_class_self_np(0x21, 0) }, 0);
            }
            #[cfg(not(target_os = "macos"))]
            fn interactive() {}
            const PASSES: usize = 8;
            // From the first thread's start to the last one's end.
            let rate = |threads: usize| {
                let mut best = f64::INFINITY;
                for _ in 0..3 {
                    let next = std::sync::atomic::AtomicUsize::new(0);
                    let start = std::sync::Barrier::new(threads);
                    let spans = std::thread::scope(|scope| {
                        let workers = (0..threads)
                            .map(|_| {
                                let (from, to, next, start) = (&from, &to, &next, &start);
                                scope.spawn(move || {
                                    interactive();
                                    start.wait();
                                    let started = std::time::Instant::now();
                                    loop {
                                        let first = next
                                            .fetch_add(64, std::sync::atomic::Ordering::Relaxed);
                                        if first >= PASSES * rows {
                                            break;
                                        }
                                        let first = first % rows;
                                        for item in (first..(first + 64).min(rows)).step_by(8) {
                                            unsafe { super::rows(from, to, item..item + 8) };
                                        }
                                    }
                                    (started, std::time::Instant::now())
                                })
                            })
                            .collect::<Vec<_>>();
                        workers
                            .into_iter()
                            .map(|worker| worker.join().unwrap())
                            .collect::<Vec<_>>()
                    });
                    let first = spans.iter().map(|span| span.0).min().unwrap();
                    let last = spans.iter().map(|span| span.1).max().unwrap();
                    best = best.min((last - first).as_secs_f64());
                }
                (PASSES * source.len()) as f64 / best / 1e9
            };
            eprintln!(
                "{:>18} -> {:<18} 1 thread {:6.2} GB/s, {threads} threads {:6.2} GB/s",
                conversion.source(),
                conversion.target(),
                rate(1),
                rate(threads)
            );
        }
    }
}
