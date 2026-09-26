// Shared pieces of the routed (mixture-of-experts) CPU entries, built on the
// projection library: expert row offsets in expert-stacked weights, the
// paired gate/up expansion with the routed SiLU . mul epilogue, the live-row
// count of a grouped block and the router's top-K selection. The counterpart
// of `metal/lib/routed/routed.h`, `cuda/lib/routed/routed.cuh` and
// `vulkan/lib/routed/routed.glsl`.
//
// Expert-stacked weights: an [E, N, K] tensor is a `Weights` operand of E * N
// rows, so expert e's row n is row e * N + n.
//
// Numerics (routed.seismic): the expansion accumulates gate and up in F32 and
// publishes `silu(gate) * up` rounded once to the activation element; down
// projections are published to it before they are weighted.

use super::super::core::{activation, reduce};
use super::super::projection::projection;
use seismic::cpu::quant::Q8Block;
use seismic::cpu::{Dense, Weights};

/// First row of expert `expert` in an [E, N, K] tensor of `rows` = N rows per
/// expert.
#[inline(always)]
pub fn expert_row(expert: i32, rows: usize) -> usize {
    usize::try_from(expert).expect("a routed expert index is non-negative") * rows
}

/// The expansion of weight rows `first..first + out.len()` (at most
/// `projection::MAX_ROWS`) of a paired gate/up operand against the F32
/// activation row `x` (or its quantized form `q8`):
/// `out[i] = A(silu(gate_i) * up_i)`, as published F32.
#[inline(always)]
pub fn expand<A: Dense>(
    gate: &Weights<'_>,
    up: &Weights<'_>,
    first: usize,
    x: &[f32],
    q8: Option<&[Q8Block]>,
    out: &mut [f32],
) {
    let mut up_sums = [0.0f32; projection::MAX_ROWS];
    let up_sums = &mut up_sums[..out.len()];
    projection::project_arithmetic(gate, first, x, q8, out);
    projection::project_arithmetic(up, first, x, q8, up_sums);
    for (target, up) in out.iter_mut().zip(up_sums.iter()) {
        *target = activation::publish::<A>(activation::silu(*target) * up);
    }
}

/// [`expand`] stored to a row span of `A` elements.
#[inline(always)]
pub fn expand_into<A: Dense>(
    gate: &Weights<'_>,
    up: &Weights<'_>,
    first: usize,
    x: &[f32],
    q8: Option<&[Q8Block]>,
    out: &mut [A::Storage],
) {
    let mut values = [0.0f32; projection::MAX_ROWS];
    let values = &mut values[..out.len()];
    expand::<A>(gate, up, first, x, q8, values);
    activation::store::<A>(values, out);
}

/// The live rows of a grouped block: its `order` row is one expert's source
/// rows followed by -1 padding, so the count is the index of the first -1.
#[inline(always)]
pub fn block_rows(order: &[i32]) -> usize {
    order.partition_point(|row| *row >= 0)
}

/// The route of one row from its router logits: the softmax over every
/// expert, then K rounds of the most probable remaining expert (ties to the
/// higher index), rank r stored at slot K - 1 - r; with `normalize` the
/// scores are divided by their slot-order sum. `probabilities` is E values of
/// work space.
#[inline(always)]
pub fn select(
    logits: &[f32],
    probabilities: &mut [f32],
    normalize: bool,
    routes: &mut [i32],
    scores: &mut [f32],
) {
    let maximum = reduce::max(logits);
    let probabilities = &mut probabilities[..logits.len()];
    for (probability, logit) in probabilities.iter_mut().zip(logits) {
        *probability = (logit - maximum).exp();
    }
    let total = reduce::sum(probabilities);
    for probability in probabilities.iter_mut() {
        *probability /= total;
    }
    let k = routes.len();
    for rank in 0..k {
        let mut winner = 0;
        for (expert, probability) in probabilities.iter().enumerate().skip(1) {
            if *probability >= probabilities[winner] {
                winner = expert;
            }
        }
        routes[k - 1 - rank] = winner as i32;
        scores[k - 1 - rank] = probabilities[winner];
        probabilities[winner] = f32::NEG_INFINITY;
    }
    if normalize {
        let denominator = scores.iter().fold(0.0f32, |sum, score| sum + score);
        for score in scores.iter_mut() {
            *score /= denominator;
        }
    }
}
