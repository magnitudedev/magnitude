// The projection library of the CPU entries: the operand prologues (the RMS
// normalization of a residual row, the staging of an activation row as F32)
// and the row-block driver over a weight operand's dot components. The
// counterpart of `metal/lib/projection/projection.h`,
// `cuda/lib/projection/projection.cuh` and
// `vulkan/lib/projection/projection.glsl`, with the same vocabulary.
//
// Decomposition: a projection launch gives each work item a block of weight
// rows (at most `MAX_ROWS`), which it projects against every activation row
// of the call, `MAX_ACTIVATIONS` at a time, or `TILED_ACTIVATIONS` as tiles:
// with several quantized rows, the tile kernel decodes each weight block once
// for all of them (`Weights::tile_q8`).
// Activation rows are staged as F32, and quantized, once per call by a
// separate launch, never per work item; or formed and quantized once by the
// kernel that produces them, in its epilogue (`emit`), where a graph pairs
// the two (`projection quantized for cpu`'s operands).
//
// Numerics: the portable `linear` accumulates `x[k] * w[k]` in F32; the dot
// components accumulate the same products in a fixed lane order, the
// reassociation every native backend is qualified under.

use super::super::core::{activation, reduce};
use seismic::cpu::components::ROW_BLOCKS;
use seismic::cpu::quant::{self, Q8Block, Q8Tile};
use seismic::cpu::{Dense, Scratch, Tensor, Weights};
use std::ops::Range;
use std::sync::atomic::{AtomicU32, Ordering};

/// The largest weight-row block of one work item (`ROWS`): an AMX tile
/// pair's sixteen rows (`form(W, INT8_AMX)`).
pub const MAX_ROWS: usize = 16;

/// The largest component row block (`ROW_BLOCKS`): an `mma16` tile's
/// (`Weights::row_block`).
pub const MAX_BLOCK: usize = ROW_BLOCKS[ROW_BLOCKS.len() - 1];

/// Quantized activation rows as a stage launch leaves them for a projection
/// ([`quantize_row`], [`quantized`]): each row's blocks, which the one-row
/// dot takes; or, for several rows over weights that all have the tile
/// form ([`tiled`]), their tiles, which the tile kernel takes.
#[derive(Clone, Copy)]
pub enum Quantized<'a> {
    Rows(&'a [Q8Block]),
    Tiles(&'a [Q8Tile]),
}

const _: () = assert!(TILED_ACTIVATIONS % quant::TILE_ROWS == 0, "a staged chunk of tiled rows is whole rows of tiles");

/// The least L1 data cache of the cores the CPU tiers serve: 32 KiB
/// (Skylake's and Zen 4's; every AMX core's is 48 KiB).
const L1_DATA_BYTES: usize = 32 << 10;

/// The activation rows one tile-kernel call takes. The call decodes each
/// weight block once for all of them (`Weights::tile_q8`), so more rows
/// amortize the decode, and the AMX core's B tiles, further. What bounds
/// them is what the call keeps in L1 across its blocks: each row's sums of
/// up to [`MAX_ROWS`] weight rows, added to block after block, which take
/// half of the least L1 ([`L1_DATA_BYTES`]), the decoded block and the
/// strip being multiplied the other half. The rows' codes, whose bytes `K`
/// sets, stream through a call once and are not held, so the bound is the
/// same at every `K` and every load, and is no tunable. (The VNNI core over
/// sixteen rows sweeps a call's codes once per eight; the `ROWS` tunable
/// chooses eight rows where that second sweep misses L2.)
pub const TILED_ACTIVATIONS: usize = L1_DATA_BYTES / 2 / (MAX_ROWS * size_of::<f32>());

/// The activation rows of a staged chunk of `quantized` rows: tile-kernel
/// calls' rows, else [`MAX_ACTIVATIONS`].
#[inline(always)]
fn chunk_rows(quantized: &Option<Quantized<'_>>) -> usize {
    if matches!(quantized, Some(Quantized::Tiles(_))) {
        TILED_ACTIVATIONS
    } else {
        MAX_ACTIVATIONS
    }
}

/// Whether `m` rows that `weights` share are quantized as tiles: several
/// rows, every weight with the tile form.
#[inline(always)]
pub fn tiled<W: Recipe + ?Sized>(m: usize, weights: &[&W]) -> bool {
    m > 1 && weights.iter().all(|weights| weights.tiled())
}

/// Quantizes staged row `row`, `values`, into `scratch` from byte `at`
/// (`ceil_div(m, Q8_ROWS) * Q8_ROWS` rows' blocks): as tiles with `tiles`
/// (its rows' [`tiled`]), else as blocks, the form [`quantized`] reads.
///
/// # Safety
/// The bytes lie in the scratch, and this work item alone writes row
/// `row`'s.
#[inline(always)]
pub unsafe fn quantize_row(values: &[f32], row: usize, tiles: bool, scratch: Scratch<'_>, at: usize) {
    let blocks = quant::blocks(values.len());
    if !tiles {
        return quant::quantize(values, unsafe { scratch.slice_mut::<Q8Block>(at + row * blocks * quant::Q8_BYTES, blocks) });
    }
    let tiles = scratch.pointer::<Q8Tile>(at + row / quant::TILE_ROWS * blocks * size_of::<Q8Tile>());
    for (b, values) in values.chunks(quant::Q8_BLOCK).enumerate() {
        let mut block = [Q8Block::ZERO];
        quant::quantize(values, &mut block);
        // SAFETY: row `row` of the tile is this work item's alone.
        unsafe { Q8Tile::place(tiles.add(b), row % quant::TILE_ROWS, &block[0]) };
    }
}

/// The `m` quantized rows of `k` values [`quantize_row`] wrote into
/// `scratch` from byte `at`, as tiles with `tiles`.
///
/// # Safety
/// The stage launch wrote them before this launch, in that form.
#[inline(always)]
pub unsafe fn quantized<'a>(scratch: Scratch<'a>, at: usize, m: usize, k: usize, tiles: bool) -> Quantized<'a> {
    let blocks = quant::blocks(k);
    if tiles {
        Quantized::Tiles(unsafe { scratch.slice(at, quant::tiles(m, blocks)) })
    } else {
        Quantized::Rows(unsafe { scratch.slice(at, m * blocks) })
    }
}

/// The arrival of `arrived` stored columns of row `row` of a result whose
/// operand the producer emits (`emits (result N over (ROWS, WIDTH) ..)`,
/// rows of `width` values): the row's last arrival forms the operand from
/// the whole stored row as a consumer's formation launch forms it from that
/// tensor's row. `form(out)` writes the row's formed values into `rows`
/// (`form_input` under the operand's norm, else the stored values staged,
/// `stage`); where `quantized` (the consumer forms Q8 blocks), their Q8
/// blocks without tiles ([`quantize_row`]) follow; then, where `nvfp4`
/// (the consumer's rows are NVFP4 operands), the rows in place as NVFP4
/// operands ([`nvfp4_row`]). The same calls in the same order on the same
/// row, so the operand has the bits of the consumer's own. A paired
/// consumer always forms its rows, so the rows' word gates the call.
///
/// One counter a row (`arrivals`' word `row`) counts columns, not work
/// items, so what completes the row is independent of how a launch's items
/// split it (empty items, rows shared by heads): the arrival is a release,
/// the completing one's an acquire, and it restores the counter to zero, as
/// `core/arrive.rs`'s. A count past `width` breaks that contract and
/// panics.
///
/// # Safety
/// `rows`, `quantized` and `arrivals` are the result's `result_N_rows`,
/// `result_N_quantized` and `result_N_arrivals` over rows of `width`
/// values, and each column of the row arrives exactly once a launch, after
/// its store.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
pub unsafe fn emit(
    rows: Scratch<'_>,
    quantized: Option<Scratch<'_>>,
    nvfp4: bool,
    arrivals: Scratch<'_>,
    width: usize,
    row: usize,
    arrived: usize,
    form: impl FnOnce(&mut [f32]),
) {
    // SAFETY: the caller's: a live, aligned word of the launch's sync
    // scratch, which every column of the row counts once.
    let counter = unsafe { AtomicU32::from_ptr(arrivals.pointer::<u32>(4 * row)) };
    let step = u32::try_from(arrived).unwrap_or_else(|_| panic!("{arrived} columns arrive at a 32-bit counter"));
    let total = counter.fetch_add(step, Ordering::AcqRel) as usize + arrived;
    assert!(total <= width, "row {row} counts {total} of its {width} columns");
    if total < width {
        return;
    }
    counter.store(0, Ordering::Relaxed);
    // SAFETY: the row's last arrival alone writes its rows and blocks.
    let formed = unsafe { rows.slice_mut::<f32>(4 * row * width, width) };
    form(formed);
    if let Some(quantized) = quantized {
        unsafe { quantize_row(formed, row, false, quantized, 0) };
    }
    if nvfp4 {
        nvfp4_row(formed);
    }
}

/// The stored values of row `row` of `rows` (`[ROWS, HEADS, WIDTH]`), the
/// row's heads one after another, staged as F32 ([`stage`] of each head):
/// the formed rows [`emit`] writes of a result whose operand is each row's
/// heads.
#[inline(always)]
pub fn stage_heads<A: Dense>(rows: &Tensor<'_, A, 3>, row: usize, out: &mut [f32]) {
    let width = rows.extent(2);
    for (head, out) in out.chunks_exact_mut(width).enumerate() {
        stage::<A>(rows.row([row, head, 0]), out);
    }
}

/// Bytes of `m` quantized rows of `k` values in either form: the
/// declarations' `ceil_div(M, Q8_ROWS) * Q8_ROWS * ceil_div(K, Q8_VALUES) *
/// Q8_BYTES`.
#[inline(always)]
pub fn quantized_bytes(m: usize, k: usize) -> usize {
    m.next_multiple_of(quant::TILE_ROWS) * quant::blocks(k) * quant::Q8_BYTES
}

/// One operand the operator's formation launch forms (`<entry>_form`,
/// whose body the build generates from the declaration's formation:
/// `NativeFormation`): `rows` rows of `width` values (`groups` groups of
/// `rows / groups` rows, group-major), formed into `staged` (F32, a row
/// after another; where only some of `weights` take NVFP4 operands, their
/// NVFP4 copy after all the rows), with their Q8 blocks into `quantized`
/// where the consumer forms them, as tiles where `tiles` (each group's at
/// its own [`quantized_bytes`]). Only the rows of `window` are formed (a
/// merge set's grouped rows; else all), each whole by part 0 where `whole`
/// (an entry function's rows).
pub struct Formed<'w, 'a, W: Recipe + ?Sized = Weights<'a>> {
    pub rows: usize,
    pub width: usize,
    pub groups: usize,
    pub window: Range<usize>,
    pub whole: bool,
    pub staged: Scratch<'a>,
    pub quantized: Option<Scratch<'a>>,
    pub tiles: bool,
    pub weights: &'w [&'w W],
}

/// Work item `item` of a formation launch whose grid holds `total` rows,
/// every operand's in turn (this one's from `first`), each in parts of
/// whole Q8 blocks ([`part_columns`]): where the item's row is one of this
/// operand's window, its part's columns of the row (all of it by part 0
/// where the operand is `whole`), `values(row, columns, out,
/// shared)` writing them (the consumer's prologue, rounded to the
/// activation element), then their Q8 blocks; a row the weights take as
/// NVFP4 operands, whose scale spans it, is formed whole by part 0, its
/// blocks quantized before the transform, as the consumer's own launch
/// always did. Every value depends on its own column (and an RMS factor
/// reduced over the whole row in one order), so the parts form the row's
/// bits whatever their count.
///
/// # Safety
/// The buffers are the declaration's, and every item of the launch calls
/// this with the same operands, so each column of a row is one item's.
#[inline(always)]
pub unsafe fn form<W: Recipe + ?Sized>(
    operand: Formed<'_, '_, W>,
    first: usize,
    total: usize,
    item: [u64; 3],
    shared: &mut [u8],
    values: impl FnOnce(usize, Range<usize>, &mut [f32], &mut [u8]),
) {
    let index = item[0] as usize;
    if index < first || index >= first + operand.rows {
        return;
    }
    let (row, width) = (index - first, operand.width);
    if !operand.window.contains(&row) {
        return;
    }
    let nvfp4 = operand.weights.iter().filter(|weights| weights.nvfp4_operands()).count();
    let columns = match nvfp4 {
        0 if !operand.whole => match part_columns(item[1], width, total) {
            Some(columns) => columns,
            None => return,
        },
        _ if item[1] == 0 => 0..width,
        _ => return,
    };
    // SAFETY: these columns of the row are this item's alone.
    let staged = unsafe { operand.staged.slice_mut::<f32>(4 * (row * width + columns.start), columns.len()) };
    values(row, columns.clone(), staged, shared);
    if let Some(quantized) = operand.quantized {
        let m = operand.rows / operand.groups;
        // SAFETY: the columns' blocks of the row are this item's alone.
        unsafe { quantize_columns(staged, row % m, columns.start, width, operand.tiles, quantized, row / m * quantized_bytes(m, width)) };
    }
    if nvfp4 == operand.weights.len() {
        nvfp4_row(staged);
    } else if nvfp4 > 0 {
        // SAFETY: the copy of row `row` is this item's alone.
        let apart = unsafe { operand.staged.slice_mut::<f32>(4 * (operand.rows + row) * width, width) };
        apart.copy_from_slice(staged);
        nvfp4_row(apart);
    }
}

/// The rows `weights` take of an operand's `rows` rows of `width` values
/// in `staged` ([`Formed`]): its NVFP4 copy where these weights take NVFP4
/// operands and not every weight of the operand (`all`) does, else the
/// formed rows.
///
/// # Safety
/// The formation, or the producer the graph pairs with it, wrote them
/// before this launch.
#[inline(always)]
pub unsafe fn operand_rows<'a, W: Recipe + ?Sized>(staged: Scratch<'a>, rows: usize, width: usize, weights: &impl Recipe, all: &[&W]) -> &'a [f32] {
    let apart = weights.nvfp4_operands() && !all.iter().all(|weights| weights.nvfp4_operands());
    unsafe { staged.slice::<f32>(if apart { 4 * rows * width } else { 0 }, rows * width) }
}

/// Quantizes columns `first..first + values.len()` (whole Q8 blocks, or
/// the row's last) of row `row` of `k` values into `scratch` from byte
/// `at`, as [`quantize_row`] lays out the whole row: as tiles with `tiles`,
/// else as blocks.
///
/// # Safety
/// The bytes lie in the scratch, and this work item alone writes those
/// blocks of row `row`.
#[inline(always)]
pub unsafe fn quantize_columns(values: &[f32], row: usize, first: usize, k: usize, tiles: bool, scratch: Scratch<'_>, at: usize) {
    debug_assert!(first % quant::Q8_BLOCK == 0, "columns of whole blocks");
    let (blocks, block) = (quant::blocks(k), first / quant::Q8_BLOCK);
    if !tiles {
        let out = unsafe { scratch.slice_mut::<Q8Block>(at + (row * blocks + block) * quant::Q8_BYTES, quant::blocks(values.len())) };
        return quant::quantize(values, out);
    }
    let tiles = scratch.pointer::<Q8Tile>(at + row / quant::TILE_ROWS * blocks * size_of::<Q8Tile>());
    for (b, values) in values.chunks(quant::Q8_BLOCK).enumerate() {
        let mut quantized = [Q8Block::ZERO];
        quant::quantize(values, &mut quantized);
        // SAFETY: row `row` of the tile is this work item's alone.
        unsafe { Q8Tile::place(tiles.add(block + b), row % quant::TILE_ROWS, &quantized[0]) };
    }
}

/// The columns part `part` of a formation forms of a row of `k` values,
/// its grid holding `rows` rows: whole Q8 blocks, [`parts`] a row, so a
/// launch of few rows (a decode step's one) spreads a row's blocks over the
/// pool and a launch of many forms each row whole. `None`: the part forms
/// nothing.
#[inline(always)]
pub fn part_columns(part: u64, k: usize, rows: usize) -> Option<Range<usize>> {
    let blocks = quant::blocks(k);
    let per = blocks.div_ceil(parts(k, rows));
    let first = part as usize * per;
    (first < blocks).then(|| first * quant::Q8_BLOCK..((first + per) * quant::Q8_BLOCK).min(k))
}

/// The parts of each row of `k` values of a formation whose grid holds
/// `rows` rows: `ceil_div(K, Q8_VALUES * ROWS)`, the operator's grid's
/// second extent at its widest operand.
#[inline(always)]
pub fn parts(k: usize, rows: usize) -> usize {
    quant::blocks(k).div_ceil(rows.max(1))
}

/// Columns `columns` of row `row` of `source`, its values taken as rows of
/// `width` in storage order (a row of `[M, Q, W]` heads is `Q * W` wide), as
/// F32: what [`form`] stages of a source of the activation element.
#[inline(always)]
pub fn flat_values<S: Dense, const R: usize>(source: &Tensor<'_, S, R>, row: usize, width: usize, columns: Range<usize>, out: &mut [f32]) {
    let inner = source.extent(R - 1);
    let (mut at, end, mut written) = (row * width + columns.start, row * width + columns.end, 0);
    while at < end {
        let mut index = [0usize; R];
        index[R - 1] = at % inner;
        let mut rest = at / inner;
        for axis in (0..R - 1).rev() {
            index[axis] = rest % source.extent(axis);
            rest /= source.extent(axis);
        }
        let length = (inner - index[R - 1]).min(end - at);
        activation::widen::<S>(source.span(index, length), &mut out[written..written + length]);
        (at, written) = (at + length, written + length);
    }
}

/// `values` rounded to the activation element `A` in place: the operand of
/// a source held in another element (F32 rows a projection takes in `A`).
#[inline(always)]
pub fn publish_row<A: Dense>(values: &mut [f32]) {
    for value in values {
        *value = activation::publish::<A>(*value);
    }
}

/// Chunk `chunk` (of [`chunk_rows`] rows) of the staged rows, `x`,
/// projected against `out.len() / count` weight rows from `first` as
/// [`project_arithmetic`] projects them, its quantized rows taken from
/// `quantized`: tiles through the tile kernel.
#[inline(always)]
fn project_chunk(weights: &Weights<'_>, first: usize, x: &[f32], quantized: Option<Quantized<'_>>, chunk: usize, out: &mut [f32]) {
    let (count, blocks) = (x.len() / weights.k(), quant::blocks(weights.k()));
    let from = chunk * chunk_rows(&quantized);
    match quantized {
        Some(Quantized::Tiles(tiles)) => {
            let rows = from / quant::TILE_ROWS * blocks..quant::tiles(from + count, blocks);
            weights.tile_q8(first, &tiles[rows], count, out)
        }
        Some(Quantized::Rows(q8)) => {
            project_arithmetic(weights, first, x, Some(&q8[from * blocks..(from + count) * blocks]), out)
        }
        None => project_arithmetic(weights, first, x, None, out),
    }
}

/// The weight rows of work item `item` when every item owns `rows` of
/// `total` rows (the last item owns the remainder).
#[inline(always)]
pub fn item_rows(item: u64, rows: u64, total: usize) -> std::ops::Range<usize> {
    let first = (item * rows) as usize;
    first.min(total)..(first + rows as usize).min(total)
}

/// The weight rows of part `part` when `parts` parts split `total` rows as
/// evenly as whole rows allow (a grid of the pool's participants: each
/// part's rows differ from another's by at most one).
#[inline(always)]
pub fn part_rows(part: u64, parts: u64, total: usize) -> std::ops::Range<usize> {
    let first = |part: u64| (part as usize * total) / parts as usize;
    first(part)..first(part + 1)
}

/// [`part_rows`] over the rows of segments enumerated in order (segment `s`
/// has `totals[s]` rows): part `part`'s share of their combined rows, as the
/// segment and its rows within it for each segment the share holds rows of.
#[inline(always)]
pub fn segment_parts(part: u64, parts: u64, totals: &[usize]) -> impl Iterator<Item = (usize, std::ops::Range<usize>)> + '_ {
    let share = part_rows(part, parts, totals.iter().sum());
    let starts = totals.iter().scan(0, |start, total| {
        let first = *start;
        *start += total;
        Some(first)
    });
    totals.iter().zip(starts).enumerate().filter_map(move |(segment, (total, first))| {
        let rows = share.start.max(first)..share.end.min(first + total);
        (rows.start < rows.end).then(|| (segment, rows.start - first..rows.end - first))
    })
}

/// `rows` in consecutive blocks of at most [`MAX_ROWS`] (what
/// [`project_staged`] takes at once): a work item's share of a pool-sized
/// split, block by block. A row's projection is its own dot whatever block
/// holds it.
#[inline(always)]
pub fn row_blocks(rows: std::ops::Range<usize>) -> impl Iterator<Item = std::ops::Range<usize>> {
    let end = rows.end;
    rows.step_by(MAX_ROWS).map(move |first| first..(first + MAX_ROWS).min(end))
}

/// `out[i] = (row first + i) · x` for `out.len()` weight rows: component
/// calls of the largest row blocks that fit.
#[inline(always)]
pub fn project(weights: &Weights<'_>, first: usize, x: &[f32], out: &mut [f32]) {
    project_arithmetic(weights, first, x, None, out)
}

/// The most activation rows one dot-component call projects
/// (`project_arithmetic`): a decode step's rows, so a decode expert's rows
/// go in one call; more are projected this many at a time.
pub const MAX_ACTIVATIONS: usize = crate::constants::VERIFY_ROWS as usize;

/// Project the `m` activation rows of `x` (`m` rows of `weights.k()`
/// values, `m` = `x.len() / k`) against `out.len() / m` weight rows from
/// `first`, activation-major: `out[a n + r] = (row first + r) · x[a]`.
/// Each row is projected with either its exact F32 activations or its
/// previously quantized row (`q8`, `m` rows of `quant::blocks(k)` blocks),
/// for weights that take it (`Weights::quantized`, the quantized dot's
/// admission); others, which take NVFP4 operand rows, multiply `x`, the rows
/// their recipe forms. A declaration admits `INT8 = 1` only where a weight
/// it multiplies takes the quantized rows (`form(W, INT8)`). The weight
/// rows go in the largest component blocks that fit, up to the weights' own
/// row block (`Weights::row_block`: an `mma16` tile's sixteen rows share
/// their code bytes); with quantized rows one component call takes up to
/// [`MAX_ACTIVATIONS`] of them, decoding each weight block once for every
/// group of up to four (`Weights::dot_q8`), each row's sums the ones it has
/// alone; one quantized row's whole 16-row tiles go in one call
/// (`Weights::dot_q8_tiles`), the same sums; rows of a tile-form element all
/// at once (`Weights::spans`).
#[inline(always)]
pub fn project_arithmetic(
    weights: &Weights<'_>,
    first: usize,
    x: &[f32],
    q8: Option<&[Q8Block]>,
    out: &mut [f32],
) {
    let k = weights.k();
    let m = x.len() / k;
    assert!(m > 0 && x.len() == m * k && out.len() % m == 0, "{} activations of {k}-value rows for {} sums", x.len(), out.len());
    let (n, blocks) = (out.len() / m, quant::blocks(k));
    let q8 = q8.filter(|_| weights.quantized());
    if let Some(q8) = q8 {
        assert_eq!(q8.len(), m * blocks, "quantized blocks of {m} activation rows");
    }
    let most = weights.row_block();
    let mut done = 0;
    // One row's whole 16-row tiles in one call (`Weights::dot_q8_tiles`).
    if let Some(q8) = q8.filter(|_| m == 1 && n >= MAX_BLOCK) {
        done = n / MAX_BLOCK * MAX_BLOCK;
        weights.dot_q8_tiles(first, q8, &mut out[..done]);
    }
    // Rows of a tile-form element: every remaining row in one call for each
    // group of activation rows (`done` is above zero only for one row).
    if let Some(q8) = q8.filter(|_| done < n && weights.spans()) {
        for from in (0..m).step_by(MAX_ACTIVATIONS) {
            let rows = from..(from + MAX_ACTIVATIONS).min(m);
            let at = rows.start * n + done;
            let sums = &mut out[at..at + rows.len() * (n - done)];
            weights.dot_q8(first + done, &q8[rows.start * blocks..rows.end * blocks], sums);
        }
        done = n;
    }
    let mut sums = [0.0f32; MAX_BLOCK * MAX_ACTIVATIONS];
    while done < n {
        let block = ROW_BLOCKS
            .iter()
            .rev()
            .copied()
            .find(|block| *block <= (n - done).min(most))
            .expect("the one-row block fits any remainder");
        for from in (0..m).step_by(MAX_ACTIVATIONS) {
            let rows = from..(from + MAX_ACTIVATIONS).min(m);
            let sums = &mut sums[..rows.len() * block];
            match q8 {
                Some(q8) => weights.dot_q8(first + done, &q8[rows.start * blocks..rows.end * blocks], sums),
                None => {
                    for (a, sums) in rows.clone().zip(sums.chunks_exact_mut(block)) {
                        weights.dot(first + done, &x[a * k..(a + 1) * k], sums);
                    }
                }
            }
            for (a, sums) in rows.zip(sums.chunks_exact(block)) {
                out[a * n + done..a * n + done + block].copy_from_slice(sums);
            }
        }
        done += block;
    }
}

/// The segment of a segmented projection holding work item `item`, and the
/// item's weight rows within it, when segment `s` has `totals[s]` weight rows
/// and every item owns `rows` of them: segments enumerate their items in
/// order, `ceil_div(totals[s], rows)` each.
#[inline(always)]
pub fn segment_rows(item: u64, rows: u64, totals: &[usize]) -> (usize, std::ops::Range<usize>) {
    let mut item = item;
    for (segment, total) in totals.iter().enumerate() {
        let items = total.div_ceil(rows as usize) as u64;
        if item < items {
            return (segment, item_rows(item, rows, *total));
        }
        item -= items;
    }
    panic!("work item beyond the segments {totals:?} of {rows} rows");
}

/// Weight rows `rows` (at most `MAX_ROWS`) projected against every staged
/// activation row (`staged` holds whole rows of `weights.k()` values):
/// `publish(row, block)` receives each activation row's projections.
#[inline(always)]
pub fn project_staged(
    weights: &Weights<'_>,
    rows: std::ops::Range<usize>,
    staged: &[f32],
    mut publish: impl FnMut(usize, &[f32]),
) {
    project_staged_arithmetic(weights, rows, staged, None, &mut publish)
}

/// Project staged rows, using their quantized rows when present and the
/// weights take them ([`project_chunk`], [`chunk_rows`] rows a call).
#[inline(always)]
pub fn project_staged_arithmetic(
    weights: &Weights<'_>,
    rows: std::ops::Range<usize>,
    staged: &[f32],
    quantized: Option<Quantized<'_>>,
    mut publish: impl FnMut(usize, &[f32]),
) {
    let quantized = quantized.filter(|_| weights.quantized());
    // A chunk's outputs on the stack, as many as its rows need: a decode
    // step's rows keep the small block.
    if chunk_rows(&quantized).min(staged.len() / weights.k()) > MAX_ACTIVATIONS {
        staged_chunks::<{ MAX_ROWS * TILED_ACTIVATIONS }>(weights, rows, staged, quantized, &mut publish)
    } else {
        staged_chunks::<{ MAX_ROWS * MAX_ACTIVATIONS }>(weights, rows, staged, quantized, &mut publish)
    }
}

/// [`project_staged_arithmetic`] with chunk outputs of `BLOCK` values.
#[inline(always)]
fn staged_chunks<const BLOCK: usize>(
    weights: &Weights<'_>,
    rows: std::ops::Range<usize>,
    staged: &[f32],
    quantized: Option<Quantized<'_>>,
    publish: &mut impl FnMut(usize, &[f32]),
) {
    let (k, n) = (weights.k(), rows.len());
    let mut block = [0.0f32; BLOCK];
    let rows_per_chunk = chunk_rows(&quantized);
    for (chunk, x) in staged.chunks(rows_per_chunk * k).enumerate() {
        let (from, count) = (chunk * rows_per_chunk, x.len() / k);
        let block = &mut block[..count * n];
        project_chunk(weights, rows.start, x, quantized, chunk, block);
        for (row, projected) in block.chunks_exact(n).enumerate() {
            publish(from + row, projected);
        }
    }
}

/// [`project_staged_arithmetic`] of `rows` weight rows of two operands (a
/// gate and an up projection), each `(weights, from)` read from its own row
/// `from`, against the same staged rows: `publish(row, first, second)`
/// receives each activation row's two blocks.
#[inline(always)]
pub fn project_pair_staged_arithmetic(
    (first, first_from): (&Weights<'_>, usize),
    (second, second_from): (&Weights<'_>, usize),
    rows: usize,
    staged: &[f32],
    quantized: Option<Quantized<'_>>,
    mut publish: impl FnMut(usize, &[f32], &[f32]),
) {
    assert_eq!(first.k(), second.k(), "paired projections share their input");
    // Paired weights share their input, and so their operand recipe.
    let quantized = quantized.filter(|_| first.quantized());
    if chunk_rows(&quantized).min(staged.len() / first.k()) > MAX_ACTIVATIONS {
        pair_chunks::<{ MAX_ROWS * TILED_ACTIVATIONS }>((first, first_from), (second, second_from), rows, staged, quantized, &mut publish)
    } else {
        pair_chunks::<{ MAX_ROWS * MAX_ACTIVATIONS }>((first, first_from), (second, second_from), rows, staged, quantized, &mut publish)
    }
}

/// [`project_pair_staged_arithmetic`] with chunk outputs of `BLOCK` values.
#[inline(always)]
fn pair_chunks<const BLOCK: usize>(
    (first, first_from): (&Weights<'_>, usize),
    (second, second_from): (&Weights<'_>, usize),
    rows: usize,
    staged: &[f32],
    quantized: Option<Quantized<'_>>,
    publish: &mut impl FnMut(usize, &[f32], &[f32]),
) {
    let k = first.k();
    let (mut first_block, mut second_block) = ([0.0f32; BLOCK], [0.0f32; BLOCK]);
    let rows_per_chunk = chunk_rows(&quantized);
    for (chunk, x) in staged.chunks(rows_per_chunk * k).enumerate() {
        let (from, count) = (chunk * rows_per_chunk, x.len() / k);
        let (first_block, second_block) = (&mut first_block[..count * rows], &mut second_block[..count * rows]);
        project_chunk(first, first_from, x, quantized, chunk, first_block);
        project_chunk(second, second_from, x, quantized, chunk, second_block);
        for (row, (projected, paired)) in first_block.chunks_exact(rows).zip(second_block.chunks_exact(rows)).enumerate() {
            publish(from + row, projected, paired);
        }
    }
}

pub use super::stage::{
    form_input, form_input_columns, nvfp4_row, operand_row, rms_row, stage, Recipe,
};
