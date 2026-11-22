// Pointwise functions of the projection epilogues, evaluated in F32 exactly as
// the portable bodies write them. The counterpart of
// `metal/lib/core/functions.h`, `cuda/lib/core/functions.cuh` and
// `vulkan/lib/core/functions.glsl`.
//
// Feed-forward activation functions, by the code the entries take
// (`activation: i32`):
//   0  SiLU       x / (1 + exp(-x))
//   1  GELU-tanh  x / (1 + exp(-2u)), u = √(2/π) (x + 0.044715 x³)
//                 (0.5 x (1 + tanh u) in its stable form)
//   2  ReLU²      max(x, 0)²
// The softcap c * tanh(z / c) is written c * (2 / (1 + exp(-2z / c)) - 1).

use super::activation;
use seismic::cpu::{Dense, F32, Store};

pub const SILU: i32 = 0;
pub const GELU_TANH: i32 = 1;
pub const RELU_SQUARED: i32 = 2;

#[inline(always)]
pub fn activate(function: i32, x: f32) -> f32 {
    match function {
        GELU_TANH => {
            #[allow(clippy::excessive_precision)]
            let u = 0.797_884_560_802_865_4_f32 * (x + 0.044715 * (x * x * x));
            x / (1.0 + (-2.0 * u).exp())
        }
        RELU_SQUARED => {
            let r = x.max(0.0);
            r * r
        }
        _ => x / (1.0 + (-x).exp()),
    }
}

/// The up-only expansion of the scaled F32 accumulator `x`, A(act(A(x))), as
/// stored in A (`function` an [`activate`] code): widened, its published
/// F32.
#[inline(always)]
pub fn activated<A: Dense>(function: i32, x: f32) -> A::Storage {
    A::narrow(activated_value::<A>(function, x))
}

/// [`activated`] before it is stored: the F32 a [`seismic::cpu::Store`] of `A` rounds.
#[inline(always)]
pub fn activated_value<A: Dense>(function: i32, x: f32) -> f32 {
    activate(function, activation::publish::<A>(x))
}

/// The gated expansion `glu` (functions.seismic) of the scaled F32
/// accumulators `gate` and `up`, as published F32 (`function` an
/// [`activate`] code).
#[inline(always)]
pub fn glu<A: Dense>(function: i32, gate: f32, up: f32) -> f32 {
    activation::publish::<A>(A::widen(activated::<A>(function, gate)) * activation::publish::<A>(up))
}

/// The limit L of an expansion whose LB (`glu_limit`'s extent, functions.seismic's
/// `glu`) is `word`: the F32 of those bits, or none for 0, which no limit takes.
#[inline(always)]
pub fn glu_limit(word: u64) -> Option<f32> {
    (word != 0).then(|| f32::from_bits(u32::try_from(word).expect("LB is the bits of an F32")))
}

/// [`glu`] clamped at `limit`: the published gate to (-inf, limit] and the
/// published up to [-limit, limit], each written `x > L ? L : x` so a NaN
/// stays NaN.
#[inline(always)]
pub fn clamped_glu<A: Dense>(function: i32, limit: f32, gate: f32, up: f32) -> f32 {
    let gate = activation::publish::<A>(gate);
    let gate = if gate > limit { limit } else { gate };
    let up = activation::publish::<A>(up);
    let up = if up > limit { limit } else { up };
    let up = if up < -limit { -limit } else { up };
    activation::publish::<A>(A::widen(A::narrow(activate(function, gate))) * up)
}

/// The expansion of a row's accumulator pairs: each `(target, gate, up)`
/// stores to `target` the published F32 [`glu`] of `gate_scale * gate` and
/// `up_scale * up` ([`clamped_glu`] at `limit` where the expansion has one).
#[inline(always)]
pub fn glu_row<A: Dense, T>(
    function: i32,
    limit: Option<f32>,
    (gate_scale, up_scale): (f32, f32),
    pairs: impl Iterator<Item = (T, f32, f32)>,
    mut store: impl FnMut(T, f32),
) {
    match limit {
        None => {
            for (target, gate, up) in pairs {
                store(target, glu::<A>(function, gate_scale * gate, up_scale * up));
            }
        }
        Some(limit) => {
            for (target, gate, up) in pairs {
                store(target, clamped_glu::<A>(function, limit, gate_scale * gate, up_scale * up));
            }
        }
    }
}

/// A gated delta net's output gate on `z`, by the code the entries take
/// (`gate_function: i32`): 0 SiLU, 1 sigmoid.
pub const GATE_SIGMOID: i32 = 1;

#[inline(always)]
pub fn output_gate(function: i32, z: f32) -> f32 {
    if function == GATE_SIGMOID {
        1.0 / (1.0 + (-z).exp())
    } else {
        z / (1.0 + (-z).exp())
    }
}

const MAX_INLINE_STREAMS: usize = 8;

#[inline(always)]
fn mix_8<const ACCUMULATE: bool>(
    weights: [f32; 8],
    srcs: [&[f32]; 8],
    out: &mut [f32],
    mut finalize: impl FnMut(usize, f32) -> f32,
) {
    let len = out.len();
    let [w0, w1, w2, w3, w4, w5, w6, w7] = weights;
    let [s0, s1, s2, s3, s4, s5, s6, s7] = [
        &srcs[0][..len],
        &srcs[1][..len],
        &srcs[2][..len],
        &srcs[3][..len],
        &srcs[4][..len],
        &srcs[5][..len],
        &srcs[6][..len],
        &srcs[7][..len],
    ];
    for c in 0..len {
        let base = if ACCUMULATE { out[c] } else { 0.0 };
        let sum = (((((((base + w0 * s0[c]) + w1 * s1[c]) + w2 * s2[c]) + w3 * s3[c])
            + w4 * s4[c])
            + w5 * s5[c])
            + w6 * s6[c])
            + w7 * s7[c];
        out[c] = finalize(c, sum);
    }
}

#[inline(always)]
fn mix_fallback<const ACCUMULATE: bool>(
    weights: &[f32],
    srcs: &[&[f32]],
    out: &mut [f32],
    mut finalize: impl FnMut(usize, f32) -> f32,
) {
    let len = out.len();
    for c in 0..len {
        let mut sum = if ACCUMULATE { out[c] } else { 0.0 };
        for (&w, &s) in weights.iter().zip(srcs) {
            sum += w * s[c];
        }
        out[c] = finalize(c, sum);
    }
}

#[inline(always)]
fn mix_streams<'r>(
    streams: usize,
    weight: impl Fn(usize) -> f32,
    stream: impl Fn(usize) -> &'r [f32],
    out: &mut [f32],
    mut finalize: impl FnMut(usize, f32) -> f32,
) {
    let len = out.len();
    let mut weights = [0.0f32; MAX_INLINE_STREAMS];
    let mut srcs: [&[f32]; MAX_INLINE_STREAMS] = [&[]; MAX_INLINE_STREAMS];
    if streams <= MAX_INLINE_STREAMS {
        for i in 0..streams {
            weights[i] = weight(i);
            srcs[i] = &stream(i)[..len];
        }
        match streams {
            0 => {
                for (c, dst) in out.iter_mut().enumerate() {
                    *dst = finalize(c, 0.0);
                }
            }
            1 => {
                let w0 = weights[0];
                let s0 = &srcs[0][..len];
                for c in 0..len {
                    out[c] = finalize(c, 0.0 + w0 * s0[c]);
                }
            }
            2 => {
                let (w0, w1) = (weights[0], weights[1]);
                let (s0, s1) = (&srcs[0][..len], &srcs[1][..len]);
                for c in 0..len {
                    out[c] = finalize(c, (0.0 + w0 * s0[c]) + w1 * s1[c]);
                }
            }
            4 => {
                let (w0, w1, w2, w3) = (weights[0], weights[1], weights[2], weights[3]);
                let (s0, s1, s2, s3) = (
                    &srcs[0][..len],
                    &srcs[1][..len],
                    &srcs[2][..len],
                    &srcs[3][..len],
                );
                for c in 0..len {
                    let sum = (((0.0 + w0 * s0[c]) + w1 * s1[c]) + w2 * s2[c]) + w3 * s3[c];
                    out[c] = finalize(c, sum);
                }
            }
            8 => mix_8::<false>(weights, srcs, out, finalize),
            _ => mix_fallback::<false>(&weights[..streams], &srcs[..streams], out, finalize),
        }
        return;
    }

    for i in 0..MAX_INLINE_STREAMS {
        weights[i] = weight(i);
        srcs[i] = &stream(i)[..len];
    }
    mix_8::<false>(weights, srcs, out, |_, sum| sum);

    let mut base = MAX_INLINE_STREAMS;
    while base + MAX_INLINE_STREAMS < streams {
        for i in 0..MAX_INLINE_STREAMS {
            weights[i] = weight(base + i);
            srcs[i] = &stream(base + i)[..len];
        }
        mix_8::<true>(weights, srcs, out, |_, sum| sum);
        base += MAX_INLINE_STREAMS;
    }

    let rem = streams - base;
    for i in 0..rem {
        weights[i] = weight(base + i);
        srcs[i] = &stream(base + i)[..len];
    }
    if rem == MAX_INLINE_STREAMS {
        mix_8::<true>(weights, srcs, out, finalize);
    } else {
        mix_fallback::<true>(&weights[..rem], &srcs[..rem], out, finalize);
    }
}

/// `scatter` (functions.seismic) of columns of one row into stream
/// `stream`: each column `c` of `out` the stream's joined value from the
/// output `y(c)`, the row's `inject_width` (`IN`) inject values
/// (`inject(i)`) over `streams` (`SN`) streams, and the streams' values of
/// the same columns (`residual(s)`): the stream's own value, or (a comb after
/// the inject values, `IN > SN`) the streams mixed by `comb[., stream]` in
/// source order, plus `y`, times the stream's inject value where the row has
/// them.
#[inline(always)]
pub fn scatter<'r>(
    streams: usize,
    inject_width: usize,
    stream: usize,
    y: impl Fn(usize) -> f32,
    inject: impl Fn(usize) -> f32,
    residual: impl Fn(usize) -> &'r [f32],
    mut out: Store<'_, F32>,
) {
    out.compute(|out| {
        if inject_width > streams {
            let factor = inject(stream);
            mix_streams(
                streams,
                |from| inject(streams + from * streams + stream),
                residual,
                out,
                |column, mixed| mixed + y(column) * factor,
            );
        } else {
            let len = out.len();
            let res = &residual(stream)[..len];
            if inject_width == 0 {
                for c in 0..len {
                    out[c] = res[c] + y(c);
                }
            } else {
                let factor = inject(stream);
                for c in 0..len {
                    out[c] = res[c] + y(c) * factor;
                }
            }
        }
    });
}

/// `collapse` (streams.seismic) of one row: each column of `out` the
/// `streams` streams' values of it (`stream(s)`) summed by `weight(s)` in
/// stream order.
#[inline(always)]
pub fn collapse<'r>(
    streams: usize,
    weight: impl Fn(usize) -> f32,
    stream: impl Fn(usize) -> &'r [f32],
    mut out: Store<'_, F32>,
) {
    out.compute(|out| {
        mix_streams(streams, weight, stream, out, |_, sum| sum);
    });
}

/// `ordered_key` (sampling.seismic): `a < b` exactly when their keys are,
/// -0 and +0 one key, and every NaN the canonical NaN's key, above +inf's.
#[inline(always)]
pub fn ordered_key(value: f32) -> u32 {
    if value.is_nan() { 0xFFC0_0000 } else { finite_ordered_key(value) }
}

/// [`ordered_key`] of a value its caller holds to be no NaN (the sampling
/// shape's, which refuses a row holding one before it keys any value).
#[inline(always)]
pub fn finite_ordered_key(value: f32) -> u32 {
    let bits = (value + 0.0).to_bits();
    if bits & 0x8000_0000 != 0 { !bits } else { bits | 0x8000_0000 }
}

#[inline(always)]
pub fn softcap(cap: f32, z: f32) -> f32 {
    cap * (2.0 / (1.0 + (-2.0 * z / cap).exp()) - 1.0)
}
