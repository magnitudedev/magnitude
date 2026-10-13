// Pointwise functions of the projection epilogues, evaluated in F32 exactly as
// the portable bodies write them. The counterpart of
// `cuda/lib/core/functions.cuh`, `vulkan/lib/core/functions.glsl` and
// `cpu/lib/core/functions.rs`.
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

namespace functions {

constant constexpr int silu = 0;
constant constexpr int gelu_tanh = 1;
constant constexpr int relu_squared = 2;

inline float activate(int function, float x) {
    if (function == gelu_tanh) {
        const float u = 0.7978845608028654f * (x + 0.044715f * (x * x * x));
        return x / (1.0f + metal::exp(-2.0f * u));
    }
    if (function == relu_squared) {
        const float r = metal::max(x, 0.0f);
        return r * r;
    }
    return x / (1.0f + metal::exp(-x));
}

inline float softcap(float cap, float z) {
    return cap * (2.0f / (1.0f + metal::exp(-2.0f * z / cap)) - 1.0f);
}

// A gated delta net's output gate on z, by the code the entries take
// (`gate_function: i32`): 0 SiLU, 1 sigmoid.
constant constexpr int gate_sigmoid = 1;

inline float output_gate(int function, float z) {
    if (function == gate_sigmoid)
        return 1.0f / (1.0f + metal::exp(-z));
    return z / (1.0f + metal::exp(-z));
}

} // namespace functions
