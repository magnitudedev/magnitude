// A row norm, the one every CPU row norm computes (vision's rows, a pooled
// source's per-token LayerNorm): two-pass F32 statistics, centered or not,
// times the weight and plus the bias when bound. The counterpart of
// `metal/lib/core/norm.h`, `cuda/lib/core/norm.cuh` and
// `vulkan/lib/core/norm.glsl`.

use super::reduce;

/// The norm of one F32 row into `out`: centered (`centered`, a layer norm) or
/// root-mean-square two-pass F32 statistics, times `weight` and plus `bias`
/// when bound.
#[inline(always)]
pub fn row(x: &[f32], centered: bool, weight: Option<&[f32]>, bias: Option<&[f32]>, epsilon: f32, out: &mut [f32]) {
    let width = x.len() as f32;
    let out = &mut out[..x.len()];
    let mean = if centered { reduce::sum(x) / width } else { 0.0 };
    for (target, value) in out.iter_mut().zip(x) {
        *target = value - mean;
    }
    let inverse = 1.0 / (reduce::sum_squares(out) / width + epsilon).sqrt();
    for target in out.iter_mut() {
        *target *= inverse;
    }
    if let Some(weight) = weight {
        for (target, w) in out.iter_mut().zip(weight) {
            *target *= w;
        }
    }
    if let Some(bias) = bias {
        for (target, b) in out.iter_mut().zip(bias) {
            *target += b;
        }
    }
}
