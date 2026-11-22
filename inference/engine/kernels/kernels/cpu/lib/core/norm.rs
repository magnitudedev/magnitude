// A row norm, the one every CPU row norm computes (vision's rows, a pooled
// source's per-token LayerNorm): two-pass F32 statistics, centered or not,
// times the weight and plus the bias when bound. The counterpart of
// `metal/lib/core/norm.h`, `cuda/lib/core/norm.cuh` and
// `vulkan/lib/core/norm.glsl`.

use super::reduce;

/// `sum((values[i] - mean)^2)` in the defined lane order of
/// `seismic::cpu::reduce::sum_squares`, without an intermediate buffer.
#[inline(always)]
fn sum_squares_centered(values: &[f32], mean: f32) -> f32 {
    use seismic::cpu::reduce::{combine, LANES};
    let mut lanes = [0.0f32; LANES];
    let chunks = values.chunks_exact(LANES);
    let tail = chunks.remainder();
    for chunk in chunks {
        for lane in 0..LANES {
            let d = chunk[lane] - mean;
            lanes[lane] = d.mul_add(d, lanes[lane]);
        }
    }
    for (lane, value) in tail.iter().enumerate() {
        let d = *value - mean;
        lanes[lane] = d.mul_add(d, lanes[lane]);
    }
    combine(lanes)
}

/// The norm of one F32 row into `out`: centered (`centered`, a layer norm) or
/// root-mean-square two-pass F32 statistics, times `weight` and plus `bias`
/// when bound.
#[inline(always)]
pub fn row(x: &[f32], centered: bool, weight: Option<&[f32]>, bias: Option<&[f32]>, epsilon: f32, out: &mut [f32]) {
    let width = x.len() as f32;
    let out = &mut out[..x.len()];
    let (mean, sum_sq) = if centered {
        let mean = reduce::sum(x) / width;
        (mean, sum_squares_centered(x, mean))
    } else {
        (0.0, reduce::sum_squares(x))
    };
    let inverse = 1.0 / (sum_sq / width + epsilon).sqrt();
    match (weight, bias) {
        (Some(weight), Some(bias)) => {
            for (((target, value), w), b) in out.iter_mut().zip(x).zip(weight).zip(bias) {
                let v = if centered { *value - mean } else { *value };
                *target = v * inverse * *w + *b;
            }
        }
        (Some(weight), None) => {
            for ((target, value), w) in out.iter_mut().zip(x).zip(weight) {
                let v = if centered { *value - mean } else { *value };
                *target = v * inverse * *w;
            }
        }
        (None, Some(bias)) => {
            for ((target, value), b) in out.iter_mut().zip(x).zip(bias) {
                let v = if centered { *value - mean } else { *value };
                *target = v * inverse + *b;
            }
        }
        (None, None) => {
            for (target, value) in out.iter_mut().zip(x) {
                let v = if centered { *value - mean } else { *value };
                *target = v * inverse;
            }
        }
    }
}
