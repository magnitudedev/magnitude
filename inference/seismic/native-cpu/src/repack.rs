//! Exact conversions of external GGUF weights into resident storage: the
//! registry's registered conversions, executed from their recipes'
//! executable form ([`program`]). [`convert`] converts a whole tensor into
//! any resident layout. Into a row layout it converts by its
//! [`RowConversion`], as pieces of whole stored tiles
//! ([`RowConversion::piece`]); into resident `rows16` and `rows8` storage,
//! one source packet becomes one storage group of its row.
//!
//! A kernel converts through [`External`] and [`Packed`] views of its bound
//! tensors; [`rows`] resolves the conversion from their representations.

mod program;

use program::{threads, PacketForms, RepackExecutor, RepackProgram, BLOCK, EXECUTION_BYTES};
pub use program::{ScaledCodeRows, ScaledCodesPlacement};
use seismic_lang::ids::RepresentationId;
use seismic_lang::registry::{
    self, ConversionLayout, PackedRowLayout, RepresentationConversion, RepresentationInfo,
    RepresentationKind, RowGeometry,
};
use std::collections::HashMap;
use std::marker::PhantomData;
use std::sync::{OnceLock, RwLock};

/// Values of each row of a tile one chunk of a chunked conversion
/// (`RowConversion::chunk`) converts at least, and so the most a row of it
/// converted whole has: its rows are converted in such chunks only when
/// they are longer. Measured (`repack_throughput`, M4 Max): sizing a chunk by
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

/// One registered conversion into a row layout as a kernel resolves it.
///
/// A tensor converts as pieces: whole source rows of one matrix, from the
/// first row of a stored tile, into their stored tiles. Both are contiguous,
/// and each piece converts apart from every other ([`piece`](Self::piece)),
/// so a tensor converts across threads, or piece by piece as its source
/// bytes arrive.
#[derive(Debug)]
pub struct RowConversion {
    conversion: &'static RepresentationConversion,
    layout: &'static PackedRowLayout,
    /// Rows of a tile of the layout.
    tile: usize,
    /// The source packets of each row of a tile one chunk converts, where
    /// the layout stores every plane of a tile group-major
    /// ([`group_major`]): rows of more packets convert in such chunks.
    /// `None` where it does not (a code plane of `mma16` or `rows32`
    /// interleaves a tile's rows): rows of any length convert whole.
    chunk: Option<usize>,
    /// The programs of the conversion's executions ([`Execution`]), each
    /// compiled when first run and kept for the process: as many as the
    /// distinct row lengths and tile remainders the process converts, which
    /// a model's dimensions bound.
    programs: RwLock<HashMap<Execution, &'static RepackProgram>>,
}

/// One execution of a row conversion's program.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum Execution {
    /// `rows` whole rows of `packets` packets: whole tiles (a power of two
    /// of them, at most `RowConversion::whole_tiles`), or the rows of one
    /// last tile.
    Whole { packets: usize, rows: usize },
    /// A chunk of `packets` packets of `rows` rows of a tile, not its rows'
    /// last: every plane's shares of the chunk's groups of the tile's rows.
    Chunk { packets: usize, rows: usize },
    /// The last chunk (of chunks of `packets` packets) of `rows` rows of a
    /// tile of rows of `k` values: every plane's shares of its groups to the
    /// plane's end in the tile.
    LastChunk { packets: usize, k: usize, rows: usize },
}

/// Whether `layout` stores every plane of a tile group-major: storage group
/// `g`'s shares of the tile's rows `r` lie one after another, `(g * tile +
/// r) * bytes_per_group` bytes past the plane's first share (its share of
/// row 0's group 0). A chunk of a tile's groups is then one span of each
/// plane, the same for every chunk but for where it starts.
fn group_major(layout: &PackedRowLayout) -> bool {
    let tile = layout.tile_rows();
    let geometry = layout
        .geometry(2 * u64::from(layout.packet.group))
        .expect("a row's geometry");
    (0..layout.planes.len()).all(|plane| {
        let bytes = u64::from(layout.planes[plane].bytes_per_group);
        let share = |row, group| layout.share_byte(&geometry, row, group, plane);
        share(0, 0).is_some_and(|first| {
            (0..2).all(|group| {
                (0..tile).all(|row| share(row, group) == Some(first + (group * tile + row) * bytes))
            })
        })
    })
}

impl RowConversion {
    fn of(conversion: &'static RepresentationConversion) -> Option<Self> {
        let ConversionLayout::Rows(layout) = conversion.destination_layout else {
            return None;
        };
        let group = conversion.source_layout.logical_group as usize;
        let tile = layout.tile_rows() as u32;
        // At least `CHUNK_VALUES` values of each row, and whole blocks of
        // each plane.
        let least = CHUNK_VALUES.div_ceil(group) as u32;
        let chunk = group_major(layout).then(|| {
            (1..)
                .map(|times| times * least)
                .find(|packets| {
                    layout.planes.iter().all(|plane| {
                        ((packets * tile * plane.bytes_per_group) as usize).is_multiple_of(BLOCK)
                    })
                })
                .expect("some multiple of an execution is whole blocks") as usize
        });
        Some(Self {
            conversion,
            layout,
            tile: tile as usize,
            chunk,
            programs: RwLock::default(),
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

    /// Rows of one stored tile of the target: a piece's rows are whole
    /// tiles but at the end of their matrix.
    pub fn tile(&self) -> usize {
        self.tile
    }

    /// Bytes of one source row of `k` values.
    pub fn source_row_bytes(&self, k: usize) -> usize {
        k.div_ceil(self.group()) * self.packet_bytes()
    }

    /// Bytes of one stored row of `k` values; `None` where the target holds
    /// no such row.
    pub fn stored_row_bytes(&self, k: usize) -> Option<usize> {
        usize::try_from(self.layout.row_stride_bytes(k as u64)?).ok()
    }

    /// Bytes of one source packet.
    fn packet_bytes(&self) -> usize {
        self.conversion.source_layout.packet_size as usize
    }

    /// Converts one piece of a tensor of rows of `k` values: `source`, whole
    /// source rows of one matrix from the first row of a stored tile, into
    /// `destination`, their stored tiles. A piece whose rows are not whole
    /// tiles ends its matrix: the rows its last tile pads are zero, as is
    /// every stored byte no plane occupies. `None` when `source` is not
    /// whole rows or `destination` not their tiles' bytes.
    pub fn piece(&'static self, k: usize, source: &[u8], destination: &mut [u8]) -> Option<()> {
        let source_row = self.source_row_bytes(k);
        if source_row == 0 {
            // Rows of no values: no bytes either side.
            return (source.is_empty() && destination.is_empty()).then_some(());
        }
        if !source.len().is_multiple_of(source_row) {
            return None;
        }
        let count = source.len() / source_row;
        if destination.len() != count.div_ceil(self.tile) * self.tile * self.stored_row_bytes(k)? {
            return None;
        }
        if count == 0 {
            return Some(());
        }
        let from = External {
            base: source.as_ptr(),
            rows: count,
            k,
            id: self.conversion.source,
            matrix_rows: count,
            life: PhantomData,
        };
        let to = Packed {
            base: destination.as_mut_ptr(),
            rows: count,
            k,
            element: registry::representation_info(self.conversion.destination),
            matrix_rows: count,
            life: PhantomData,
        };
        // SAFETY: the views are exactly the borrowed slices: the source's
        // rows, and the destination's tiles, which nothing else references.
        unsafe { self::rows(&from, &to, 0..count) };
        Some(())
    }

    /// The pieces of a matrix of `rows` rows of `k` values of at most
    /// `tiles` stored tiles each, in order: each one's source bytes and
    /// stored bytes, from the matrix's first of each. `None` where the
    /// target holds no row of `k` values.
    pub fn pieces(
        &self,
        k: usize,
        rows: usize,
        tiles: std::num::NonZeroUsize,
    ) -> Option<impl Iterator<Item = (std::ops::Range<usize>, std::ops::Range<usize>)>> {
        let (tile, source_row, tiles) = (self.tile, self.source_row_bytes(k), tiles.get());
        let stride = self.stored_row_bytes(k)?;
        Some((0..rows).step_by(tiles * tile).map(move |first| {
            let end = (first + tiles * tile).min(rows);
            (
                first * source_row..end * source_row,
                first * stride..end.next_multiple_of(tile) * stride,
            )
        }))
    }

    /// The canonical bytes of a tensor of `extents` from its canonical
    /// source bytes `source`, as pieces across threads when there are many.
    /// `None` for a tensor of no packing axis.
    fn tensor(&'static self, extents: &[u64], source: &[u8]) -> Option<Vec<u8>> {
        let (&k, leading) = extents.split_last()?;
        let k = k as usize;
        let n = leading.last().map_or(1, |rows| *rows as usize);
        let matrices = leading[..leading.len().saturating_sub(1)]
            .iter()
            .product::<u64>() as usize;
        let matrix_tiles = n.div_ceil(self.tile);
        let (matrix_source, matrix_stored) = (
            n * self.source_row_bytes(k),
            matrix_tiles * self.tile * self.stored_row_bytes(k)?,
        );
        let mut output = vec![0u8; matrices * matrix_stored];
        if output.is_empty() {
            return Some(output);
        }
        let threads = threads(source.len());
        let per_thread = std::num::NonZeroUsize::new((matrices * matrix_tiles).div_ceil(threads))
            .expect("a tensor of stored bytes has tiles");
        // Each matrix's pieces of at most a thread's tiles, whose stored
        // bytes follow one another.
        let mut rest = output.as_mut_slice();
        let mut work = Vec::new();
        for matrix in 0..matrices {
            for (from, to) in self.pieces(k, n, per_thread)? {
                let (stored, tail) = std::mem::take(&mut rest).split_at_mut(to.len());
                let from = matrix * matrix_source + from.start..matrix * matrix_source + from.end;
                work.push((&source[from], stored));
                rest = tail;
            }
        }
        let per_thread = work.len().div_ceil(threads);
        std::thread::scope(|scope| {
            for pieces in work.chunks_mut(per_thread) {
                scope.spawn(move || {
                    for (source, stored) in pieces {
                        self.piece(k, source, stored)
                            .expect("a tensor's pieces are whole rows and their tiles");
                    }
                });
            }
        });
        Some(output)
    }

    /// The bytes of a tile of rows of `k` values that chunk `chunk` (of
    /// `packets` packets) writes in each plane: the shares of its groups of the
    /// tile's rows and, of the last, every byte to the plane's end (the
    /// alignment past its shares and the shares of padding groups, which hold
    /// zero).
    fn chunk_segments(
        &self,
        packets: usize,
        k: usize,
        chunk: usize,
    ) -> Vec<std::ops::Range<usize>> {
        let (layout, tile, packets) = (self.layout, self.tile as u64, packets as u64);
        let geometry = layout.geometry(k as u64).expect("a row's geometry");
        let row_packets = k.div_ceil(self.group()) as u64;
        let first = chunk as u64 * packets;
        let last = first + packets >= row_packets;
        (0..layout.planes.len())
            .map(|plane| {
                let start = layout
                    .share_byte(&geometry, 0, first, plane)
                    .expect("a group-major plane keeps its shares together");
                let end = if last {
                    geometry
                        .offsets
                        .get(plane + 1)
                        .map_or(geometry.stride, |offset| *offset)
                        * tile
                } else {
                    start + packets * tile * u64::from(layout.planes[plane].bytes_per_group)
                };
                start as usize..end as usize
            })
            .collect()
    }

    /// The program of `execution`, compiled when first run.
    fn program(&self, execution: Execution) -> &'static RepackProgram {
        if let Some(program) = self
            .programs
            .read()
            .expect("no thread panics holding the programs")
            .get(&execution)
        {
            return program;
        }
        self.programs
            .write()
            .expect("no thread panics holding the programs")
            .entry(execution)
            .or_insert_with(|| {
                Box::leak(Box::new(match execution {
                    Execution::Whole { packets, rows } => self.compile_whole_rows(packets, rows),
                    Execution::Chunk { packets, rows } => self.compile_chunk(packets, rows, None),
                    Execution::LastChunk { packets, k, rows } => {
                        self.compile_chunk(packets, rows, Some(k))
                    }
                }))
            })
    }

    /// The recipe over a chunk's packets (of `packets` packets) of each of
    /// `rows` rows of a tile,
    /// each row an input segment, into a segment per plane of the target's
    /// layout (in plane order) holding the chunk's groups of every row of
    /// the tile as the layout places them (`chunk_segments`): any chunk but
    /// a row's last, or, `last` of rows of `k` values, its last. Every byte
    /// of a segment no packet bit maps to is zero: those of the rows past
    /// `rows`, and the plane's bytes past the last chunk's shares.
    fn compile_chunk(&self, packets: usize, rows: usize, last: Option<usize>) -> RepackProgram {
        let layout = self.layout;
        // A chunk but the last converts the same at any row length: chunk 0
        // of rows two chunks long.
        let (k, chunk) = match last {
            Some(k) => (k, k.div_ceil(self.group()).div_ceil(packets) - 1),
            None => (2 * packets * self.group(), 0),
        };
        let geometry = layout.geometry(k as u64).expect("a row's geometry");
        let segments = self.chunk_segments(packets, k, chunk);
        let first = chunk * packets;
        let chunk_packets = (k.div_ceil(self.group()) - first).min(packets);
        RepackProgram::compile(
            self.conversion,
            rows as u32,
            chunk_packets as u32,
            &segments
                .iter()
                .map(|segment| segment.len() as u32)
                .collect::<Vec<_>>(),
            |row, packet, plane, bit| {
                if !layout.occupies(plane, bit) {
                    return None;
                }
                let at = layout.packet_bit(
                    &geometry,
                    u64::from(row),
                    (first + packet as usize) as u64,
                    plane,
                    bit,
                );
                let index = segments
                    .iter()
                    .position(|segment| {
                        (segment.start as u64 * 8..segment.end as u64 * 8).contains(&at)
                    })
                    .expect("a placed bit lies in a plane's segment");
                Some((index, (at - segments[index].start as u64 * 8) as u32))
            },
        )
        .expect("a tile of an execution's chunk of a row fits a program")
    }

    /// Tiles an execution of whole rows of `packets` packets converts at
    /// most ([`whole_tiles`]).
    fn whole_tiles(&self, packets: usize) -> usize {
        whole_tiles(self.tile, packets * self.packet_bytes())
    }

    /// The recipe over `rows` rows' packets, `packets` each, one input
    /// segment, into their stored tiles, one output segment: every byte no
    /// plane occupies zero, and so every byte of the rows a last tile pads.
    fn compile_whole_rows(&self, packets: usize, rows: usize) -> RepackProgram {
        let layout = self.layout;
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
            &[(rows.div_ceil(self.tile) * self.tile) as u32 * geometry.stride as u32],
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
    /// converting a whole tensor into a packet layout ([`convert`]).
    forms: OnceLock<PacketForms>,
    /// Its form when it converts into a row layout.
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
/// canonical source bytes of a tensor of `extents`: into a row layout by its
/// [`RowConversion`], into a packet layout every packet through the
/// recipe's program; across threads when there are many. `None` when
/// `source` is not the source's canonical byte count over `extents`.
pub fn convert(
    conversion: &RepresentationConversion,
    extents: &[u64],
    source: &[u8],
) -> Option<Vec<u8>> {
    if registry::canonical_bytes(conversion.source, extents)? != source.len() as u64 {
        return None;
    }
    let executors = &executors()[conversion.id.position()];
    Some(match &executors.rows {
        Some(rows) => rows.tensor(extents, source)?,
        None => executors
            .forms
            .get_or_init(|| PacketForms::compile(conversion))
            .execute(source),
    })
}

/// Every registered conversion into a row layout.
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

/// The conversion of `source` into `target`, when one is registered into a
/// row layout.
fn conversion(
    source: RepresentationId,
    target: RepresentationId,
) -> Option<&'static RowConversion> {
    executors()[registry::representation_conversion(source, target)?
        .id
        .position()]
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

/// A bound result tensor of a packed row-layout element, written only by a
/// conversion.
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
    /// Whole rows by the programs of their length ([`Whole`]).
    Whole(Whole),
    /// Chunks of each row of a tile ([`Chunks`]).
    Chunks(Box<Chunks>),
}

/// The executors of a plan's programs of whole rows, each made when first
/// run: of each power of two of whole tiles up to an execution's
/// (`RowConversion::whole_tiles`), and of each count of rows of a last
/// tile short of whole.
struct Whole {
    tiles: Vec<Option<RepackExecutor<'static>>>,
    last: Vec<Option<RepackExecutor<'static>>>,
}

/// How a chunked plan converts a tile: chunk `c` of each row writes
/// `segments[c]` of the tile, a span of each plane; the executors of its
/// programs by rows of the tile present (`rows - 1`), of any chunk but the
/// last and of the last, each made when first run; and the bases of an
/// execution in place: per row its source, per plane its output.
struct Chunks {
    packets: usize,
    segments: Vec<Vec<std::ops::Range<usize>>>,
    chunk: Vec<Option<RepackExecutor<'static>>>,
    last: Vec<Option<RepackExecutor<'static>>>,
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
        let form = match conversion.chunk {
            Some(packets) if row_packets > packets => Form::Chunks(Box::new(Chunks {
                packets,
                segments: (0..row_packets.div_ceil(packets))
                    .map(|chunk| conversion.chunk_segments(packets, source.k, chunk))
                    .collect(),
                chunk: (0..conversion.tile).map(|_| None).collect(),
                last: (0..conversion.tile).map(|_| None).collect(),
                sources: Vec::with_capacity(conversion.tile),
                outputs: vec![std::ptr::null_mut(); conversion.layout.planes.len()],
            })),
            _ => Form::Whole(Whole {
                tiles: (0..=conversion.whole_tiles(row_packets).ilog2())
                    .map(|_| None)
                    .collect(),
                last: (0..conversion.tile).map(|_| None).collect(),
            }),
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

    /// The first stored byte of the tile of `row`: in a tiled layout padded
    /// to whole tiles per matrix.
    fn stored(&self, destination: &Packed<'_>, row: usize) -> *mut u8 {
        let (tile, n) = (self.conversion.tile, destination.matrix_rows);
        let stored = row / n * n.div_ceil(tile) * tile + row % n;
        // SAFETY: the row lies in the destination (`from_tensor`).
        unsafe { destination.base.add(stored * self.stride()) }
    }

    /// Converts `rows` by executions of whole rows: of whole tiles, each
    /// the most a power of two of them up to an execution's, then of the
    /// rows of a last tile short of whole.
    ///
    /// # Safety
    /// As [`rows`]'s.
    unsafe fn whole(
        &self,
        executions: &mut Whole,
        source: &External<'_>,
        destination: &Packed<'_>,
        rows: std::ops::Range<usize>,
    ) {
        let (conversion, source_row) = (self.conversion, self.source_row);
        let tile = conversion.tile;
        let source_bytes = source.rows * source_row;
        let most = executions.tiles.len() - 1;
        let mut first = rows.start;
        while first < rows.end {
            let left = rows.end - first;
            let (present, executor) = if left >= tile {
                let tiles = (left / tile).ilog2().min(most as u32) as usize;
                let executor = executions.tiles[tiles].get_or_insert_with(|| {
                    conversion
                        .program(Execution::Whole {
                            packets: self.row_packets,
                            rows: tile << tiles,
                        })
                        .executor()
                });
                (tile << tiles, executor)
            } else {
                let executor = executions.last[left].get_or_insert_with(|| {
                    conversion
                        .program(Execution::Whole {
                            packets: self.row_packets,
                            rows: left,
                        })
                        .executor()
                });
                (left, executor)
            };
            // SAFETY: (all below) this work item owns the rows, and their
            // entire tiles; the stored tiles of an execution's rows follow
            // one another.
            let (input, output) = (
                unsafe { source.base.add(first * source_row) },
                self.stored(destination, first),
            );
            // In place when the source holds a block past the rows.
            if (first + present) * source_row + BLOCK <= source_bytes {
                unsafe { executor.run_into(&[input], &[output]) };
            } else {
                executor.source().copy_from_slice(unsafe {
                    std::slice::from_raw_parts(input, present * source_row)
                });
                let placed = executor.run();
                unsafe { std::slice::from_raw_parts_mut(output, placed.len()) }
                    .copy_from_slice(placed);
            }
            first += present;
        }
    }

    /// Converts `rows` tile by tile, each row in chunks ([`Chunks`]).
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
            packets,
            segments,
            chunk,
            last,
            sources,
            outputs,
        } = chunks;
        let packets = *packets;
        let conversion = self.conversion;
        let (tile, source_row, k) = (conversion.tile, self.source_row, source.k);
        let source_bytes = source.rows * source_row;
        let (execution, count) = (packets * conversion.packet_bytes(), segments.len());
        for first in rows.clone().step_by(tile) {
            let present = (rows.end - first).min(tile);
            // SAFETY: this work item owns the rows, and their entire tile,
            // including its padded rows.
            let out = unsafe {
                std::slice::from_raw_parts_mut(
                    self.stored(destination, first),
                    tile * self.stride(),
                )
            };
            for (index, segments) in segments.iter().enumerate() {
                let start = index * execution;
                let bytes = execution.min(source_row - start);
                let executor = if index + 1 < count {
                    chunk[present - 1].get_or_insert_with(|| {
                        conversion
                            .program(Execution::Chunk { packets, rows: present })
                            .executor()
                    })
                } else {
                    last[present - 1].get_or_insert_with(|| {
                        conversion
                            .program(Execution::LastChunk { packets, k, rows: present })
                            .executor()
                    })
                };
                // In place when the source holds a block past the chunk's
                // packets of the last present row.
                if (first + present - 1) * source_row + start + bytes + BLOCK <= source_bytes {
                    sources.clear();
                    // SAFETY: each present row's packets of the chunk and a
                    // block past them lie in the source.
                    sources.extend(
                        (first..first + present)
                            .map(|row| unsafe { source.base.add(row * source_row + start) }),
                    );
                    // Every plane's base derives from one pointer to `out`,
                    // which nothing else references until the run is done.
                    let tile_base = out.as_mut_ptr();
                    for (base, segment) in outputs.iter_mut().zip(segments) {
                        *base = tile_base.wrapping_add(segment.start);
                    }
                    // SAFETY: as above; the planes' segments lie in `out`,
                    // each apart from the others.
                    unsafe { executor.run_into(sources, outputs) };
                    continue;
                }
                for (r, input) in executor.source().chunks_exact_mut(bytes).enumerate() {
                    // SAFETY: the row lies inside the source (`from_tensor`).
                    input.copy_from_slice(unsafe {
                        std::slice::from_raw_parts(
                            source.base.add((first + r) * source_row + start),
                            bytes,
                        )
                    });
                }
                let placed = executor.run();
                let mut at = 0;
                for segment in segments {
                    out[segment.clone()].copy_from_slice(&placed[at..at + segment.len()]);
                    at += segment.len();
                }
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
    use seismic_lang::registry::{pseudo_random_bytes, Layout};

    /// Canonical source bytes of `rows` rows of `k` values for `conversion`,
    /// in the conversion's domain (its scales converted as numbers finite:
    /// NaN payloads are not part of the conversion contract).
    fn admitted_source(conversion: &RowConversion, rows: usize, k: usize, seed: u64) -> Vec<u8> {
        let mut source = pseudo_random_bytes(seed, rows * conversion.source_row_bytes(k));
        conversion.conversion.admit_source(&mut source);
        source
    }

    /// The first byte at which `out` differs from the registry's `expected`
    /// bytes of rows of `stride` bytes, described.
    fn difference(
        conversion: &RowConversion,
        k: usize,
        stride: usize,
        out: &[u8],
        expected: &[u8],
    ) -> Option<String> {
        if out.len() != expected.len() {
            return Some(format!(
                "{} bytes, the registry's {}",
                out.len(),
                expected.len()
            ));
        }
        let at = out.iter().zip(expected).position(|(a, b)| a != b)?;
        Some(format!(
            "{} -> {} k {k}: byte {at} (row {}, offset {}) is {:#04x}, the registry's {:#04x}",
            conversion.source(),
            conversion.target(),
            at / stride,
            at % stride,
            out[at],
            expected[at],
        ))
    }

    /// Every conversion into a row layout reproduces the registry's host
    /// repack bit for bit, on rows whose last packet is partial, staged and
    /// in place: through kernel work items of six tiles (some ending on an
    /// execution of fewer tiles than the most, in place or staged, or on a
    /// matrix's last tile short of whole), through [`convert`], and as
    /// pieces of every size from one tile, converted last first; over rows
    /// of fewer and more packets than a chunked execution's.
    #[test]
    fn conversions_match_the_registry_repack() {
        let mut checked = 0;
        for conversion in conversions() {
            let execution = conversion
                .chunk
                .map_or(CHUNK_VALUES, |packets| packets * conversion.group());
            for (rows, k) in [
                (3, conversion.group()),
                (3, 3 * conversion.group()),
                (3, 2560),
                (3, 2560 + 40),
                (17, 2560 + 40),
                (11, 2 * execution + conversion.group() + 40),
            ] {
                if conversion.stored_row_bytes(k).is_none() {
                    continue;
                }
                let source = admitted_source(conversion, rows, k, k as u64);
                let expected =
                    seismic_lang::interp::repack(conversion.conversion.id, &[rows, k], &source)
                        .expect("registry repack");
                let stride = conversion.stored_row_bytes(k).unwrap();
                let mut out = vec![0xa5u8; rows.next_multiple_of(conversion.tile) * stride];
                let from = unsafe {
                    External::from_tensor(
                        source.as_ptr(),
                        &[rows as u64, k as u64],
                        &[
                            conversion
                                .source_row_bytes(k)
                                .div_ceil(conversion.packet_bytes())
                                as u64,
                            1,
                        ],
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
                let item = 6 * conversion.tile;
                for first in (0..rows).step_by(item) {
                    unsafe { super::rows(&from, &to, first..(first + item).min(rows)) };
                }
                if let Some(difference) = difference(conversion, k, stride, &out, &expected) {
                    panic!("work items: {difference}");
                }
                let converted = convert(conversion.conversion, &[rows as u64, k as u64], &source)
                    .expect("a conversion of canonical bytes");
                if let Some(difference) = difference(conversion, k, stride, &converted, &expected) {
                    panic!("convert: {difference}");
                }
                for tiles in 1..=rows.div_ceil(conversion.tile) {
                    let mut pieced = vec![0x5au8; expected.len()];
                    let tiles = std::num::NonZeroUsize::new(tiles).unwrap();
                    let pieces = conversion.pieces(k, rows, tiles).unwrap().collect::<Vec<_>>();
                    for (from, to) in pieces.into_iter().rev() {
                        conversion
                            .piece(k, &source[from], &mut pieced[to])
                            .expect("a piece of whole rows into their tiles");
                    }
                    if let Some(difference) = difference(conversion, k, stride, &pieced, &expected)
                    {
                        panic!("pieces of {tiles} tiles: {difference}");
                    }
                }
                checked += 1;
            }
        }
        assert!(checked > 100, "{checked} conversions and shapes checked");
    }

    /// A tensor of several matrices whose rows are not whole tiles converts
    /// as the registry repacks it: each matrix padded to whole tiles.
    #[test]
    fn stacked_matrices_match_the_registry_repack() {
        for conversion in conversions() {
            let k = 4 * conversion.group().max(64);
            let Some(stride) = conversion.stored_row_bytes(k) else {
                continue;
            };
            let (matrices, rows) = (3, 2 * conversion.tile + 3);
            let source = admitted_source(conversion, matrices * rows, k, 7);
            let extents = [matrices as u64, rows as u64, k as u64];
            let expected = seismic_lang::interp::repack(
                conversion.conversion.id,
                &[matrices, rows, k],
                &source,
            )
            .expect("registry repack");
            let converted = convert(conversion.conversion, &extents, &source).unwrap();
            if let Some(difference) = difference(conversion, k, stride, &converted, &expected) {
                panic!("{difference}");
            }
        }
    }

    /// A piece that is not whole rows into their tiles is refused.
    #[test]
    fn pieces_are_whole_rows_into_their_tiles() {
        let conversion = resolve("gguf_q4_k", "q4k@mma16").expect("q4_k into mma16");
        let (k, tile) = (512, conversion.tile());
        let (row, stride) = (
            conversion.source_row_bytes(k),
            conversion.stored_row_bytes(k).unwrap(),
        );
        let source = admitted_source(conversion, tile + 1, k, 3);
        let mut tiles = vec![0u8; 2 * tile * stride];
        assert!(conversion
            .piece(k, &source[..row * tile + 1], &mut tiles[..tile * stride])
            .is_none());
        assert!(conversion
            .piece(k, &source[..row * tile], &mut tiles[..tile * stride - 16])
            .is_none());
        // A piece one row into a second tile writes that whole tile.
        assert!(conversion
            .piece(k, &source, &mut tiles[..tile * stride])
            .is_none());
        assert!(conversion.piece(k, &source, &mut tiles).is_some());
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
            let source = admitted_source(conversion, rows, k, 0);
            let mut out = vec![0u8; rows * conversion.stored_row_bytes(k).unwrap()];
            let from = unsafe {
                External::from_tensor(
                    source.as_ptr(),
                    &[rows as u64, k as u64],
                    &[k.div_ceil(conversion.group()) as u64, 1],
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
            const PASSES: usize = 8;
            // Items of whole tiles of at least eight rows.
            let item = conversion.tile.max(8);
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
                                        for item_first in
                                            (first..(first + 64).min(rows)).step_by(item)
                                        {
                                            unsafe {
                                                super::rows(from, to, item_first..item_first + item)
                                            };
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

    /// This thread at the highest quality of service (on the performance
    /// cores, where macOS schedules it).
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

    /// Source bytes of one piece of an expert record a landing converts as
    /// its bytes arrive: the 128 KiB read unit.
    const PIECE_BYTES: usize = 128 << 10;

    /// The host's conversion of one expert record into CUDA's `mma16`, as a
    /// file miss lands it: its tensors' rows of the Qwen3.6 35B-A3B
    /// shapes (gate and up `[512, 2048]`, down `[2048, 512]`), in each
    /// GGUF source's `mma16` storage, converted piece by piece, each piece
    /// the most whole tiles of at most [`PIECE_BYTES`] source bytes. Per
    /// source: GB/s of source bytes on one thread (a record converted whole,
    /// median of passes), the median and worst µs a piece takes, and the GB/s
    /// each thread reaches when every available thread converts records of
    /// its own at once. The q4k/q4k/q5k record of the model's Q4 is reported
    /// last, as a whole.
    #[test]
    #[ignore = "measurement; run with --release --ignored --nocapture"]
    fn mma16_record_rate() {
        let threads = std::thread::available_parallelism().unwrap().get();
        const PASSES: usize = 64;
        /// The pieces of a `[rows, k]` tensor of `conversion`: source and
        /// destination ranges.
        fn pieces(
            conversion: &RowConversion,
            rows: usize,
            k: usize,
        ) -> Vec<(std::ops::Range<usize>, std::ops::Range<usize>)> {
            let tiles = PIECE_BYTES / (conversion.tile * conversion.source_row_bytes(k));
            conversion
                .pieces(k, rows, std::num::NonZeroUsize::new(tiles.max(1)).unwrap())
                .unwrap()
                .collect()
        }
        /// A record: per tensor its conversion, rows and k, and source.
        type Record = Vec<(&'static RowConversion, usize, usize, Vec<u8>)>;
        let record = |tensors: &[(&'static RowConversion, usize, usize)], seed: u64| -> Record {
            tensors
                .iter()
                .map(|(conversion, rows, k)| {
                    (
                        *conversion,
                        *rows,
                        *k,
                        admitted_source(conversion, *rows, *k, seed),
                    )
                })
                .collect()
        };
        let source_bytes = |record: &Record| {
            record
                .iter()
                .map(|(_, _, _, source)| source.len())
                .sum::<usize>()
        };
        // Seconds to convert `record` into `out`, piece by piece; each
        // piece's seconds pushed onto `pieced`.
        let convert_record = |record: &Record, out: &mut [Vec<u8>], pieced: &mut Vec<f64>| {
            let started = std::time::Instant::now();
            for ((conversion, rows, k, source), out) in record.iter().zip(out.iter_mut()) {
                for (from, to) in pieces(conversion, *rows, *k) {
                    let piece = std::time::Instant::now();
                    conversion.piece(*k, &source[from], &mut out[to]).unwrap();
                    pieced.push(piece.elapsed().as_secs_f64());
                }
            }
            started.elapsed().as_secs_f64()
        };
        let outputs = |record: &Record| -> Vec<Vec<u8>> {
            record
                .iter()
                .map(|(conversion, rows, k, _)| {
                    vec![
                        0u8;
                        rows.next_multiple_of(conversion.tile)
                            * conversion.stored_row_bytes(*k).unwrap()
                    ]
                })
                .collect()
        };
        let median = |values: &mut Vec<f64>| {
            values.sort_by(f64::total_cmp);
            values[values.len() / 2]
        };
        let measure = |name: &str, tensors: &[(&'static RowConversion, usize, usize)]| {
            interactive();
            let one = record(tensors, 1);
            let mut out = outputs(&one);
            // Equal to the registry's conversion, and warm.
            convert_record(&one, &mut out, &mut Vec::new());
            for ((conversion, rows, k, source), out) in one.iter().zip(&out) {
                let expected =
                    seismic_lang::interp::repack(conversion.conversion.id, &[*rows, *k], source)
                        .unwrap();
                assert!(
                    out == &expected,
                    "{name}: {} -> {} differs from the registry",
                    conversion.source(),
                    conversion.target()
                );
            }
            let (mut whole, mut pieced) = (Vec::new(), Vec::new());
            for _ in 0..PASSES {
                whole.push(convert_record(&one, &mut out, &mut pieced));
            }
            let bytes = source_bytes(&one) as f64;
            let alone = bytes / median(&mut whole) / 1e9;
            let piece = median(&mut pieced) * 1e6;
            let worst = pieced.last().copied().unwrap() * 1e6;
            // Every thread converting records of its own at once.
            let start = std::sync::Barrier::new(threads);
            let rates = std::thread::scope(|scope| {
                let workers = (0..threads)
                    .map(|index| {
                        let start = &start;
                        scope.spawn(move || {
                            interactive();
                            let records = (0..4)
                                .map(|seed| record(tensors, 16 + (index * 4 + seed) as u64))
                                .collect::<Vec<_>>();
                            let mut outs = records.iter().map(outputs).collect::<Vec<_>>();
                            start.wait();
                            let started = std::time::Instant::now();
                            for pass in 0..PASSES {
                                let at = pass % records.len();
                                convert_record(&records[at], &mut outs[at], &mut Vec::new());
                            }
                            (PASSES * source_bytes(&records[0])) as f64
                                / started.elapsed().as_secs_f64()
                                / 1e9
                        })
                    })
                    .collect::<Vec<_>>();
                workers
                    .into_iter()
                    .map(|worker| worker.join().unwrap())
                    .collect::<Vec<_>>()
            });
            let least = rates.iter().copied().fold(f64::INFINITY, f64::min);
            let mean = rates.iter().sum::<f64>() / rates.len() as f64;
            eprintln!(
                "{name:>42}: {:.3} MB, 1 thread {alone:6.2} GB/s, piece median {piece:6.1} µs worst {worst:6.1} µs, \
                 {threads} threads {mean:6.2} GB/s each (least {least:6.2})",
                bytes / 1e6
            );
        };
        let (h, f) = (2048, 512);
        for conversion in conversions() {
            if registry::representation_info(conversion.conversion.destination).layout
                != Layout::Mma16
            {
                continue;
            }
            let name = format!("{} -> {}", conversion.source(), conversion.target());
            measure(
                &name,
                &[(conversion, f, h), (conversion, f, h), (conversion, h, f)],
            );
        }
        let q4 = resolve("gguf_q4_k", "q4k@mma16").expect("q4_k into mma16");
        let q5 = resolve("gguf_q5_k", "q5k@mma16").expect("q5_k into mma16");
        measure(
            "35B-A3B Q4 record (q4k, q4k, q5k)",
            &[(q4, f, h), (q4, f, h), (q5, h, f)],
        );
    }
}
