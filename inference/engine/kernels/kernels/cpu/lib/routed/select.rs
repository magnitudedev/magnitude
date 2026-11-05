// Expert selection of the general routed feed-forward (`routed_select`), one
// row at a time: scores, the selection-only bias, top-K and the combine
// weights. The counterpart of `metal/lib/routed/select.h`,
// `cuda/lib/routed/select.cuh` and `vulkan/lib/routed/select.glsl`.
//
// Top-K is one sort of the row's experts by `ranked::rank`: its first K
// entries are the portable body's K argmax rounds in order.

use super::ranked;

pub const SOFTMAX: i32 = 0;
pub const SIGMOID: i32 = 1;

pub const SUM: i32 = 1;
pub const SUM_PLUS_EPSILON: i32 = 2;
pub const CLAMPED_SUM: i32 = 3;

/// The row's scores from its logits (in place): softmax over the row,
/// sigmoid, or sqrt(softplus).
#[inline(always)]
pub fn scores(function: i32, values: &mut [f32]) {
    match function {
        SOFTMAX => {
            let maximum = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let mut total = 0.0f32;
            for value in values.iter_mut() {
                *value = (*value - maximum).exp();
                total += *value;
            }
            for value in values.iter_mut() {
                *value /= total;
            }
        }
        SIGMOID => {
            for value in values.iter_mut() {
                *value = 1.0 / (1.0 + (-*value).exp());
            }
        }
        _ => {
            for value in values.iter_mut() {
                *value = (value.max(0.0) + (1.0 + (-value.abs()).exp()).ln()).sqrt();
            }
        }
    }
}

/// The weight of a selected score after normalization over the slot-order
/// sum `denominator`.
#[inline(always)]
pub fn normalized(normalization: i32, weight: f32, denominator: f32, epsilon: f32) -> f32 {
    match normalization {
        SUM => weight / denominator,
        SUM_PLUS_EPSILON => weight / (denominator + epsilon),
        CLAMPED_SUM => weight / denominator.max(epsilon),
        _ => weight,
    }
}

/// Selects K = `routes.len()` experts of one row from its `scores` and the
/// selection `bias`: ranked on score + bias (`ranked::rank`), rank r stored
/// at slot K - 1 - r with its unbiased score in `weights`. `order` is E
/// scratch indices.
#[inline(always)]
pub fn top_k(scores: &[f32], bias: impl Fn(usize) -> f32, order: &mut [u32], routes: &mut [i32], weights: &mut [f32]) {
    ranked::rank(order, |expert| scores[expert] + bias(expert));
    let k = routes.len();
    for (rank, expert) in order[..k].iter().enumerate() {
        routes[k - 1 - rank] = *expert as i32;
        weights[k - 1 - rank] = scores[*expert as usize];
    }
}
