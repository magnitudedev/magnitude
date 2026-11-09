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
use seismic::cpu::Dense;

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
    A::narrow(activate(function, activation::publish::<A>(x)))
}

/// The gated expansion `glu` (functions.seismic) of the scaled F32
/// accumulators `gate` and `up`, as published F32 (`function` an
/// [`activate`] code).
#[inline(always)]
pub fn glu<A: Dense>(function: i32, gate: f32, up: f32) -> f32 {
    activation::publish::<A>(A::widen(activated::<A>(function, gate)) * activation::publish::<A>(up))
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

/// `scatter` (functions.seismic) of one column of one row: stream `stream`'s
/// joined value from the output `y`, the row's `inject_width` (`IN`) inject
/// values (`inject(i)`) over `streams` (`SN`) streams, and the streams'
/// values of the column (`residual(s)`): the stream's own value, or (a comb
/// after the inject values, `IN > SN`) the streams mixed by `comb[., stream]`
/// in source order, plus `y`, times the stream's inject value where the row
/// has them.
#[inline(always)]
pub fn scatter(
    streams: usize,
    inject_width: usize,
    stream: usize,
    y: f32,
    inject: impl Fn(usize) -> f32,
    residual: impl Fn(usize) -> f32,
) -> f32 {
    let mixed = if inject_width > streams {
        (0..streams).fold(0.0f32, |mixed, from| mixed + inject(streams + from * streams + stream) * residual(from))
    } else {
        residual(stream)
    };
    if inject_width == 0 { mixed + y } else { mixed + y * inject(stream) }
}

#[inline(always)]
pub fn softcap(cap: f32, z: f32) -> f32 {
    cap * (2.0 / (1.0 + (-2.0 * z / cap).exp()) - 1.0)
}
