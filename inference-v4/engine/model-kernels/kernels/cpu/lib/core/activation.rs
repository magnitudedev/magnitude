// The engine's activation convention on CPU: the element `A` an entry binds
// (`bf16`, `f16` or `f32`) is where the portable bodies publish intermediate
// values, so every value an entry stores to an `A` tensor is the F32
// computation rounded once to `A`. The counterpart of
// `metal/lib/core/activation.h` (`element::Act`), `cuda/lib/core/activation.cuh`
// and `vulkan/lib/core/activation.glsl`.

use seismic::cpu::Dense;

/// `value` as the portable body publishes it to element `E`.
#[inline(always)]
pub fn publish<E: Dense>(value: f32) -> f32 {
    E::round(value)
}

/// `silu(value)` = `value / (1 + exp(-value))`, the portable `silu`.
#[inline(always)]
pub fn silu(value: f32) -> f32 {
    value / (1.0 + (-value).exp())
}

/// `sigmoid(value)` = `1 / (1 + exp(-value))`, the portable `sigmoid`.
#[inline(always)]
pub fn sigmoid(value: f32) -> f32 {
    1.0 / (1.0 + (-value).exp())
}

/// The F32 values of a row of `E` elements.
#[inline(always)]
pub fn widen<E: Dense>(row: &[E::Storage], out: &mut [f32]) {
    seismic::cpu::tensor::widen_row::<E>(row, out)
}

/// F32 values stored to a row of `E` elements (each rounded once).
#[inline(always)]
pub fn store<E: Dense>(values: &[f32], row: &mut [E::Storage]) {
    seismic::cpu::tensor::narrow_row::<E>(values, row)
}
