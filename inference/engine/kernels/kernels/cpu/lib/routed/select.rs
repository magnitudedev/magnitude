// Expert selection of the general routed feed-forward (`routed_select`), one
// row at a time: scores, the selection-only bias, top-K and the combine
// weights. The counterpart of `metal/lib/routed/select.h`,
// `cuda/lib/routed/select.cuh` and `vulkan/lib/routed/select.glsl`.
//
// Top-K is one stable sort of the row's (ranked, expert) pairs by descending
// ranked value, ties to the higher expert index: its first K entries are the
// portable body's K argmax rounds in order.

use std::cmp::Ordering;

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

/// The first `count` `u32` values of a work item's private bytes (which the
/// pool aligns for any scalar): the selection's expert order.
#[inline(always)]
pub fn indices(bytes: &mut [u8], count: usize) -> &mut [u32] {
    assert!(bytes.len() >= 4 * count, "{count} indices need {} private bytes", 4 * count);
    // SAFETY: every bit pattern is a `u32`; `align_to_mut` places only aligned
    // whole values in the middle part.
    let (head, middle, _) = unsafe { bytes.align_to_mut::<u32>() };
    assert!(head.is_empty(), "private bytes are aligned for u32");
    &mut middle[..count]
}

/// Selects K = `routes.len()` experts of one row from its `scores` and the
/// selection `bias`: ranked on score + bias (ties to the higher expert), rank
/// r stored at slot K - 1 - r with its unbiased score in `weights`.
/// `order` is E scratch indices.
#[inline(always)]
pub fn top_k(scores: &[f32], bias: impl Fn(usize) -> f32, order: &mut [u32], routes: &mut [i32], weights: &mut [f32]) {
    for (index, slot) in order.iter_mut().enumerate() {
        *slot = index as u32;
    }
    let ranked = |expert: u32| scores[expert as usize] + bias(expert as usize);
    order.sort_by(|a, b| {
        ranked(*b)
            .partial_cmp(&ranked(*a))
            .unwrap_or(Ordering::Equal)
            .then(b.cmp(a))
    });
    let k = routes.len();
    for (rank, expert) in order[..k].iter().enumerate() {
        routes[k - 1 - rank] = *expert as i32;
        weights[k - 1 - rank] = scores[*expert as usize];
    }
}
