// How every expert selection orders its experts (`rank_experts`,
// functions.seismic): the route's `routed::select` and `select::top_k`. The
// counterpart of `metal/lib/routed/ranked.h`, `cuda/lib/routed/ranked.cuh`
// and `vulkan/lib/routed/ranked.glsl`.
//   ranked: a NaN as -inf.
//   rank: descending ranked value, ties to the higher expert index. Over
//     ranked values it is a total order, so every rank names an expert
//     whatever a row holds.

use std::cmp::Ordering;

#[inline(always)]
pub fn ranked(value: f32) -> f32 {
    if value.is_nan() { f32::NEG_INFINITY } else { value }
}

/// The first `count` `u32` values of a work item's private bytes (which the
/// pool aligns for any scalar): a selection's expert order.
#[inline(always)]
pub fn indices(bytes: &mut [u8], count: usize) -> &mut [u32] {
    assert!(bytes.len() >= 4 * count, "{count} indices need {} private bytes", 4 * count);
    // SAFETY: every bit pattern is a `u32`; `align_to_mut` places only aligned
    // whole values in the middle part.
    let (head, middle, _) = unsafe { bytes.align_to_mut::<u32>() };
    assert!(head.is_empty(), "private bytes are aligned for u32");
    &mut middle[..count]
}

/// Orders the E expert indices of `order` by their ranked `value`: its first
/// K entries are the portable body's K winners in rank order.
#[inline(always)]
pub fn rank(order: &mut [u32], value: impl Fn(usize) -> f32) {
    for (index, slot) in order.iter_mut().enumerate() {
        *slot = index as u32;
    }
    order.sort_unstable_by(|a, b| {
        let (a_value, b_value) = (ranked(value(*a as usize)), ranked(value(*b as usize)));
        if a_value > b_value {
            Ordering::Less
        } else if a_value < b_value {
            Ordering::Greater
        } else {
            b.cmp(a)
        }
    });
}
