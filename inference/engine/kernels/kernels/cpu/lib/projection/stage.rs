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

/// A staged row as the products over `weights`, which share it, take it: in
/// place, as NVFP4 operands where the weights take them (`nvfp4_row`), else
/// as it is.
#[inline(always)]
pub fn operand_row(weights: &[&Weights<'_>], row: &mut [f32]) {
    let nvfp4 = weights[0].nvfp4_operands();
    assert!(
        weights.iter().all(|weights| weights.nvfp4_operands() == nvfp4),
        "weights that share rows share their operand recipe"
    );
    if nvfp4 {
        nvfp4_row(row)
    }
}

/// A row as NVFP4 operands (docs/precision.md, "NVFP4 operands"; the portable
/// `nvfp4_row`), in place: g, the least power of two at or above
/// amax(|row|) / (6 * 448); per 16-value block s, amax(|block|) / (6 * g)
/// onto UE4M3; each value q * g * s with q, x / (g * s) onto E2M1, saturating
/// at 6 (zero where s is). Every step rounds to nearest even, and q * g * s
/// is exact in F32.
pub fn nvfp4_row(row: &mut [f32]) {
    let amax = row.iter().fold(0.0f32, |held, value| held.max(value.abs()));
    let bits = (amax / 2688.0).to_bits();
    let g = f32::from_bits(((bits >> 23) + u32::from(bits & 0x7f_ffff != 0)).clamp(1, 254) << 23);
    for block in row.chunks_mut(16) {
        let maximum = block.iter().fold(0.0f32, |held, value| held.max(value.abs()));
        let d = g * ue4m3(maximum / (6.0 * g));
        for value in block {
            *value = if d == 0.0 { 0.0 } else { e2m1(*value / d) * d };
        }
    }
}

/// v >= 0 onto UE4M3 (exponent bias 7, three mantissa bits, subnormal
/// quantum 2^-9), saturating at 448.
fn ue4m3(v: f32) -> f32 {
    let exponent = ((v.to_bits() >> 23) & 255).max(121);
    let quantum = f32::from_bits((exponent - 3) << 23);
    ((v / quantum).round_ties_even() * quantum).min(448.0)
}

/// v onto E2M1's magnitudes with v's sign, saturating at 6; NaN stays NaN.
fn e2m1(v: f32) -> f32 {
    let a = v.abs();
    let m = if a <= 0.25 {
        0.0
    } else if a < 0.75 {
        0.5
    } else if a <= 1.25 {
        1.0
    } else if a < 1.75 {
        1.5
    } else if a <= 2.5 {
        2.0
    } else if a < 3.5 {
        3.0
    } else if a <= 5.0 {
        4.0
    } else if a > 5.0 {
        6.0
    } else {
        a
    };
    if v < 0.0 { -m } else { m }
}
