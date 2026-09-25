// The projection library of the CPU entries: the operand prologues (the RMS
// normalization of a residual row, the staging of an activation row as F32)
// and the row-block driver over a weight operand's dot components. The
// counterpart of `metal/lib/projection/projection.h`,
// `cuda/lib/projection/projection.cuh` and
// `vulkan/lib/projection/projection.glsl`, with the same vocabulary.
//
// Decomposition: a projection launch gives each work item a block of weight
// rows (at most 8, one dot-component call per activation row), which it
// projects against every activation row of the call. The block's weights stay
// in cache across the activation rows, and activation rows are staged as F32
// once per call by a separate launch, never per work item.
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

/// Scratch bytes per quantized activation block. Keep the Seismic scratch
/// declarations in sync with this representation.
pub const Q8_BYTES: usize = std::mem::size_of::<Q8Block>();

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

/// Project with either exact F32 activations or one previously quantized row.
#[inline(always)]
pub fn project_arithmetic(
    weights: &Weights<'_>,
    first: usize,
    x: &[f32],
    q8: Option<&[Q8Block]>,
    out: &mut [f32],
) {
    let mut done = 0;
    while done < out.len() {
        let block = ROW_BLOCKS
            .iter()
            .rev()
            .copied()
            .find(|block| *block <= out.len() - done)
            .expect("the one-row block fits any remainder");
        if let Some(q8) = q8 {
            weights.dot_q8(first + done, q8, &mut out[done..done + block]);
        } else {
            weights.dot(first + done, x, &mut out[done..done + block]);
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

/// Project staged rows, using their corresponding Q8 rows when present.
#[inline(always)]
pub fn project_staged_arithmetic(
    weights: &Weights<'_>,
    rows: std::ops::Range<usize>,
    staged: &[f32],
    quantized: Option<&[Q8Block]>,
    mut publish: impl FnMut(usize, &[f32]),
) {
    let mut completed = 0;
    if rows.len() == 8 {
        if let Some(q8) = quantized {
            let blocks = quant::blocks(weights.k());
            let total = staged.len() / weights.k();
            while completed + 4 <= total {
                let mut tile = [0.0f32; 32];
                weights.gemm_q8(rows.start, &q8[completed * blocks..(completed + 4) * blocks], &mut tile);
                for m in 0..4 {
                    publish(completed + m, &tile[m * 8..(m + 1) * 8]);
                }
                completed += 4;
            }
        }
    }
    let mut block = [0.0f32; MAX_ROWS];
    let block = &mut block[..rows.len()];
    for (row, x) in staged.chunks_exact(weights.k()).enumerate().skip(completed) {
        let q8 = quantized.map(|blocks| {
            let stride = quant::blocks(weights.k());
            &blocks[row * stride..(row + 1) * stride]
        });
        project_arithmetic(weights, rows.start, x, q8, block);
        publish(row, block);
    }
}

pub use super::stage::{normalize, quantize, rms_row, stage};
