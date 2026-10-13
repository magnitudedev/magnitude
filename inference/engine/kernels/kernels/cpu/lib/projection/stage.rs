use super::super::core::{activation, reduce};
use seismic::cpu::quant::{self, Q8Block};
use seismic::cpu::{Dense, Weights};

#[inline(always)]
pub fn quantize(row: &[f32], out: &mut [Q8Block]) {
    quant::quantize(row, out)
}

/// The RMS normalization of residual row `x` with the norm operand `norm`
/// (decoded into the work item's private bytes `shared`), each value
/// published to `A` as `rms_row` publishes it.
#[inline(always)]
pub fn normalize<A: Dense>(
    x: &[f32],
    norm: &Weights<'_>,
    epsilon: f32,
    shared: &mut [u8],
    out: &mut [f32],
) {
    let weight = seismic::cpu::tensor::floats(shared, x.len());
    norm.decode_row(0, weight);
    rms_row::<A>(x, weight, epsilon, out);
}

/// A projection entry's input row (`form_input` in functions.seismic): the
/// RMS normalization of residual row `x` with `norm` when the entry has one,
/// else the sum in part order of the parts of `x`, rows of `out.len()`
/// values a hyper-connection mix site publishes; each value published to `A`.
#[inline(always)]
pub fn form_input<A: Dense>(
    x: &[f32],
    norm: Option<&Weights<'_>>,
    epsilon: f32,
    shared: &mut [u8],
    out: &mut [f32],
) {
    match norm {
        Some(norm) => normalize::<A>(x, norm, epsilon, shared, out),
        None => sum_parts::<A>(x, out),
    }
}

/// The sum in part order of the `out.len()`-wide parts of `x`, each value
/// published to `A`.
#[inline(always)]
pub fn sum_parts<A: Dense>(x: &[f32], out: &mut [f32]) {
    let width = out.len();
    for (column, target) in out.iter_mut().enumerate() {
        let mut sum = 0.0f32;
        for part in x.chunks_exact(width) {
            sum += part[column];
        }
        *target = activation::publish::<A>(sum);
    }
}

/// The RMS normalization of residual row `x` with the decoded norm `weight`,
/// each value published to `A`: `A(x * rsqrt(sum(x^2) / len + epsilon) * w)`.
#[inline(always)]
pub fn rms_row<A: Dense>(x: &[f32], weight: &[f32], epsilon: f32, out: &mut [f32]) {
    let inverse = reduce::rms_inverse(x, epsilon);
    let out = &mut out[..x.len()];
    for ((value, w), target) in x.iter().zip(&weight[..x.len()]).zip(out) {
        *target = activation::publish::<A>(value * inverse * w);
    }
}

/// An activation row of `A` elements staged as F32.
#[inline(always)]
pub fn stage<A: Dense>(row: &[A::Storage], out: &mut [f32]) {
    activation::widen::<A>(row, out)
}
