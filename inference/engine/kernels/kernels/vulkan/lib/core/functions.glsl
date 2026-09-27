// Pointwise functions of the projection epilogues, evaluated in F32 exactly as
// the portable bodies write them. The counterpart of
// `metal/lib/core/functions.h`, `cuda/lib/core/functions.cuh` and
// `cpu/lib/core/functions.rs`. Divisions round correctly
// (`seismic_div_rn`); GELU and the softcap use the bounded-error `exp` (their
// arguments are unbounded), SiLU the plain one as the Qwen SiLU does.
//
// Feed-forward activation functions, by the code the entries take
// (`activation: i32`):
//   0  SiLU       x / (1 + exp(-x))
//   1  GELU-tanh  x / (1 + exp(-2u)), u = √(2/π) (x + 0.044715 x³)
//                 (0.5 x (1 + tanh u) in its stable form)
//   2  ReLU²      max(x, 0)²
// The softcap c * tanh(z / c) is written c * (2 / (1 + exp(-2z / c)) - 1).
//
// This file is independent of any entry ABI.
#include "precise.glsl"

#define FUNCTIONS_SILU 0
#define FUNCTIONS_GELU_TANH 1
#define FUNCTIONS_RELU_SQUARED 2

float functions_activate(const int function, float x) {
    if (function == FUNCTIONS_GELU_TANH) {
        const float u = 0.7978845608028654 * (x + 0.044715 * (x * x * x));
        return seismic_div_rn(x, 1.0 + precise_exp(-2.0 * u));
    }
    if (function == FUNCTIONS_RELU_SQUARED) {
        const float r = max(x, 0.0);
        return r * r;
    }
    return seismic_div_rn(x, 1.0 + exp(-x));
}

float functions_softcap(float cap, float z) {
    return cap * (seismic_div_rn(2.0, 1.0 + precise_exp(seismic_div_rn(-2.0 * z, cap))) - 1.0);
}
