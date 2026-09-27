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

#[inline(always)]
pub fn softcap(cap: f32, z: f32) -> f32 {
    cap * (2.0 / (1.0 + (-2.0 * z / cap).exp()) - 1.0)
}
