// The projection library of the CPU entries: the operand prologues (the RMS
// normalization of a residual row, the staging of an activation row as F32)
// and the row-block driver over a weight operand's dot components. The
// counterpart of `metal/lib/projection/projection.h`,
// `cuda/lib/projection/projection.cuh` and
// `vulkan/lib/projection/projection.glsl`, with the same vocabulary.
//
// Decomposition: a projection launch gives each work item a block of weight
// rows (at most 8), which it projects against every activation row of the
// call, `MAX_ACTIVATIONS` at a time: with several quantized rows, the tile
// kernel decodes each weight block once for all of them (`Weights::tile_q8`).
// Activation rows are staged as F32, and quantized, once per call by a
// separate launch, never per work item.
//
// Numerics: the portable `linear` accumulates `x[k] * w[k]` in F32; the dot
// components accumulate the same products in a fixed lane order, the
// reassociation every native backend is qualified under.

use super::super::core::{activation, reduce};
use seismic::cpu::components::ROW_BLOCKS;
use seismic::cpu::quant::{self, Q8Block, Q8Tile};
use seismic::cpu::{Dense, Scratch, Weights};

/// The largest weight-row block of one work item.
pub const MAX_ROWS: usize = 8;

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

const _: () = assert!(MAX_ACTIVATIONS == quant::TILE_ROWS, "a staged chunk of rows is one row of tiles");

/// Whether `m` rows that `weights` share are quantized as tiles: several
/// rows, every weight with the tile form.
#[inline(always)]
pub fn tiled(m: usize, weights: &[&Weights<'_>]) -> bool {
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

/// Bytes of `m` quantized rows of `k` values in either form: the
/// declarations' `ceil_div(M, Q8_ROWS) * Q8_ROWS * ceil_div(K, Q8_VALUES) *
/// Q8_BYTES`.
#[inline(always)]
pub fn quantized_bytes(m: usize, k: usize) -> usize {
    m.next_multiple_of(quant::TILE_ROWS) * quant::blocks(k) * quant::Q8_BYTES
}

/// Chunk `chunk` (of [`MAX_ACTIVATIONS`] rows) of the staged rows, `x`,
/// projected against `out.len() / count` weight rows from `first` as
/// [`project_arithmetic`] projects them, its quantized rows taken from
/// `quantized`: tiles through the tile kernel.
#[inline(always)]
fn project_chunk(weights: &Weights<'_>, first: usize, x: &[f32], quantized: Option<Quantized<'_>>, chunk: usize, out: &mut [f32]) {
    let (count, blocks) = (x.len() / weights.k(), quant::blocks(weights.k()));
    match quantized {
        Some(Quantized::Tiles(tiles)) => weights.tile_q8(first, &tiles[chunk * blocks..(chunk + 1) * blocks], count, out),
        Some(Quantized::Rows(q8)) => {
            let from = chunk * MAX_ACTIVATIONS;
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

/// `out[i] = (row first + i) · x` for `out.len()` weight rows: component
/// calls of the largest row blocks that fit.
#[inline(always)]
pub fn project(weights: &Weights<'_>, first: usize, x: &[f32], out: &mut [f32]) {
    project_arithmetic(weights, first, x, None, out)
}

/// The most activation rows one dot-component call projects
/// (`project_arithmetic`): a decode step's rows, so a decode expert's rows
/// go in one call; more are projected this many at a time.
pub const MAX_ACTIVATIONS: usize = crate::constants::DECODE_ROWS as usize;

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
/// weights take them ([`project_chunk`], [`MAX_ACTIVATIONS`] rows a call).
#[inline(always)]
pub fn project_staged_arithmetic(
    weights: &Weights<'_>,
    rows: std::ops::Range<usize>,
    staged: &[f32],
    quantized: Option<Quantized<'_>>,
    mut publish: impl FnMut(usize, &[f32]),
) {
    let quantized = quantized.filter(|_| weights.quantized());
    let (k, n) = (weights.k(), rows.len());
    let mut block = [0.0f32; MAX_ROWS * MAX_ACTIVATIONS];
    for (chunk, x) in staged.chunks(MAX_ACTIVATIONS * k).enumerate() {
        let (from, count) = (chunk * MAX_ACTIVATIONS, x.len() / k);
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
    let k = first.k();
    let (mut first_block, mut second_block) = ([0.0f32; MAX_ROWS * MAX_ACTIVATIONS], [0.0f32; MAX_ROWS * MAX_ACTIVATIONS]);
    for (chunk, x) in staged.chunks(MAX_ACTIVATIONS * k).enumerate() {
        let (from, count) = (chunk * MAX_ACTIVATIONS, x.len() / k);
        let (first_block, second_block) = (&mut first_block[..count * rows], &mut second_block[..count * rows]);
        project_chunk(first, first_from, x, quantized, chunk, first_block);
        project_chunk(second, second_from, x, quantized, chunk, second_block);
        for (row, (projected, paired)) in first_block.chunks_exact(rows).zip(second_block.chunks_exact(rows)).enumerate() {
            publish(from + row, projected, paired);
        }
    }
}

pub use super::stage::{
    form_input, normalize, nvfp4_row, operand_row, quantize, rms_row, stage, sum_parts,
};
