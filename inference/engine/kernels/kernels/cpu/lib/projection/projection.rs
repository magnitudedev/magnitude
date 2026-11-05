// The projection library of the CPU entries: the operand prologues (the RMS
// normalization of a residual row, the staging of an activation row as F32)
// and the row-block driver over a weight operand's dot components. The
// counterpart of `metal/lib/projection/projection.h`,
// `cuda/lib/projection/projection.cuh` and
// `vulkan/lib/projection/projection.glsl`, with the same vocabulary.
//
// Decomposition: a projection launch gives each work item a block of weight
// rows (at most 8), which it projects against every activation row of the
// call, `MAX_ACTIVATIONS` at a time: with quantized rows, one dot-component
// call decodes each weight block once for every group of up to four of them.
// Activation rows are staged as F32 once per call by a separate launch, never
// per work item.
//
// Numerics: the portable `linear` accumulates `x[k] * w[k]` in F32; the dot
// components accumulate the same products in a fixed lane order, the
// reassociation every native backend is qualified under.

use super::super::core::{activation, reduce};
use seismic::cpu::components::ROW_BLOCKS;
use seismic::cpu::quant::{self, Q8Block};
use seismic::cpu::{Dense, Weights};

/// The largest weight-row block of one work item.
pub const MAX_ROWS: usize = 8;

/// The largest component row block (`ROW_BLOCKS`): an `mma16` tile's
/// (`Weights::row_block`).
pub const MAX_BLOCK: usize = ROW_BLOCKS[ROW_BLOCKS.len() - 1];

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
/// alone.
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
    // One row's whole tiles of a 16-row block layout in one call
    // (`Weights::dot_q8_tiles`).
    if let Some(q8) = q8.filter(|_| m == 1 && most == MAX_BLOCK && n >= MAX_BLOCK) {
        done = n / MAX_BLOCK * MAX_BLOCK;
        weights.dot_q8_tiles(first, q8, &mut out[..done]);
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

/// Project staged rows, using their corresponding Q8 rows when present and
/// the weights take them (`project_arithmetic`, [`MAX_ACTIVATIONS`] rows a
/// call).
#[inline(always)]
pub fn project_staged_arithmetic(
    weights: &Weights<'_>,
    rows: std::ops::Range<usize>,
    staged: &[f32],
    quantized: Option<&[Q8Block]>,
    mut publish: impl FnMut(usize, &[f32]),
) {
    let quantized = quantized.filter(|_| weights.quantized());
    let (k, blocks, n) = (weights.k(), quant::blocks(weights.k()), rows.len());
    let mut block = [0.0f32; MAX_ROWS * MAX_ACTIVATIONS];
    for (chunk, x) in staged.chunks(MAX_ACTIVATIONS * k).enumerate() {
        let (from, count) = (chunk * MAX_ACTIVATIONS, x.len() / k);
        let q8 = quantized.map(|q8| &q8[from * blocks..(from + count) * blocks]);
        let block = &mut block[..count * n];
        project_arithmetic(weights, rows.start, x, q8, block);
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
    quantized: Option<&[Q8Block]>,
    mut publish: impl FnMut(usize, &[f32], &[f32]),
) {
    assert_eq!(first.k(), second.k(), "paired projections share their input");
    // Paired weights share their input, and so their operand recipe.
    let quantized = quantized.filter(|_| first.quantized());
    let (k, blocks) = (first.k(), quant::blocks(first.k()));
    let (mut first_block, mut second_block) = ([0.0f32; MAX_ROWS * MAX_ACTIVATIONS], [0.0f32; MAX_ROWS * MAX_ACTIVATIONS]);
    for (chunk, x) in staged.chunks(MAX_ACTIVATIONS * k).enumerate() {
        let (from, count) = (chunk * MAX_ACTIVATIONS, x.len() / k);
        let q8 = quantized.map(|q8| &q8[from * blocks..(from + count) * blocks]);
        let (first_block, second_block) = (&mut first_block[..count * rows], &mut second_block[..count * rows]);
        project_arithmetic(first, first_from, x, q8, first_block);
        project_arithmetic(second, second_from, x, q8, second_block);
        for (row, (projected, paired)) in first_block.chunks_exact(rows).zip(second_block.chunks_exact(rows)).enumerate() {
            publish(from + row, projected, paired);
        }
    }
}

pub use super::stage::{
    form_input, normalize, nvfp4_row, operand_row, quantize, rms_row, stage, sum_parts,
};
