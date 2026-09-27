// Reductions of the CPU entries, in the defined order of
// `seismic::cpu::reduce` (fixed lanes, fixed combine tree), so every tier
// gives the same bits. The counterpart of `metal/lib/core/reduce.h`,
// `cuda/lib/core/reduce.cuh` and `vulkan/lib/core/reduce.glsl`.

pub use seismic::cpu::reduce::{argmax, dot, max, sum, sum_squares};

/// `1 / sqrt(sum(x^2) / len + epsilon)`: the RMS normalization factor of a
/// row. The portable bodies sum the squares in any order.
#[inline(always)]
pub fn rms_inverse(x: &[f32], epsilon: f32) -> f32 {
    1.0 / (sum_squares(x) / x.len() as f32 + epsilon).sqrt()
}
