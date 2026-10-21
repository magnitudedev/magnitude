// Expert selection of the general routed feed-forward (`routed_select`), one
// warp per row: scores, the selection-only bias, top-K and the combine
// weights. The counterpart of `metal/lib/routed/select.h`,
// `vulkan/lib/routed/select.glsl` and `cpu/lib/routed/select.rs`.
//
// Lane `lane` holds experts lane + 32 i (i < P = ceil(E / 32)); when E is
// not a multiple of 32, lanes at or past E % 32 hold P - 1 (`candidates`)
// and their last register is an excluded entry (ranked -inf, expert -1, as
// a popped one). Top-K is one pass over a lane-local order: every lane sorts
// its P candidates once (an odd-even transposition network over registers),
// then each of the K rounds is a single warp argmax over the 32 lane heads
// (ties to the higher expert index) and one shift of the winning lane's
// candidates. The winners are the portable body's K argmax rounds in the
// same order.
//
// This file is independent of any entry ABI.

#include "../core/reduce.cuh"

namespace select {

constexpr int softmax = 0;
constexpr int sigmoid = 1;
constexpr int sqrt_softplus = 2;

constexpr int no_normalization = 0;
constexpr int sum = 1;
constexpr int sum_plus_epsilon = 2;
constexpr int clamped_sum = 3;

// The experts lane `lane` holds of E: ceil(E / 32) on the first E % 32
// lanes, E / 32 on the others (every lane E / 32 when 32 divides E).
template <unsigned E> __device__ __forceinline__ unsigned candidates(unsigned lane) {
    return E / 32 + (E % 32 != 0 && lane < E % 32 ? 1u : 0u);
}

// Descending value, ties to the higher expert index.
__device__ __forceinline__ bool precedes(float value, int expert, float best, int best_expert) {
    return value > best || (value == best && expert > best_expert);
}

// The scores of this lane's experts from their logits (in place); the
// softmax runs over the `held` leading values only (the others are excluded
// entries, whose scores are unused).
template <unsigned P> __device__ __forceinline__ void scores(int function, float (&values)[P], unsigned held) {
    if (function == softmax) {
        float maximum = -__int_as_float(0x7f800000);
#pragma unroll
        for (unsigned i = 0; i < P; ++i)
            if (i < held)
                maximum = fmaxf(maximum, values[i]);
        maximum = seismic_warp_max_f32(maximum);
        float total = 0.0f;
#pragma unroll
        for (unsigned i = 0; i < P; ++i) {
            if (i < held) {
                values[i] = expf(values[i] - maximum);
                total = seismic_add_rn(total, values[i]);
            }
        }
        total = seismic_warp_sum_f32(total);
#pragma unroll
        for (unsigned i = 0; i < P; ++i)
            values[i] = values[i] / total;
    } else if (function == sigmoid) {
#pragma unroll
        for (unsigned i = 0; i < P; ++i)
            values[i] = 1.0f / (1.0f + expf(-values[i]));
    } else {
#pragma unroll
        for (unsigned i = 0; i < P; ++i)
            values[i] = sqrtf(fmaxf(values[i], 0.0f) + logf(1.0f + expf(-fabsf(values[i]))));
    }
}

// Sorts this lane's (ranked, score, expert) triples by `precedes`.
template <unsigned P>
__device__ __forceinline__ void sort(float (&ranked)[P], float (&score)[P], int (&expert)[P]) {
#pragma unroll
    for (unsigned pass = 0; pass < P; ++pass) {
#pragma unroll
        for (unsigned i = pass & 1u; i + 1u < P; i += 2u) {
            if (precedes(ranked[i + 1u], expert[i + 1u], ranked[i], expert[i])) {
                const float r = ranked[i], s = score[i];
                const int e = expert[i];
                ranked[i] = ranked[i + 1u];
                score[i] = score[i + 1u];
                expert[i] = expert[i + 1u];
                ranked[i + 1u] = r;
                score[i + 1u] = s;
                expert[i + 1u] = e;
            }
        }
    }
}

// Pops the lane's head (the winning lane only).
template <unsigned P> __device__ __forceinline__ void pop(float (&ranked)[P], float (&score)[P], int (&expert)[P]) {
#pragma unroll
    for (unsigned i = 0; i + 1u < P; ++i) {
        ranked[i] = ranked[i + 1u];
        score[i] = score[i + 1u];
        expert[i] = expert[i + 1u];
    }
    ranked[P - 1] = -__int_as_float(0x7f800000);
    expert[P - 1] = -1;
}

// The weight of a selected score after normalization over the slot-order
// sum `denominator`.
__device__ __forceinline__ float normalized(int normalization, float weight, float denominator, float epsilon) {
    if (normalization == sum)
        return weight / denominator;
    if (normalization == sum_plus_epsilon)
        return weight / seismic_add_rn(denominator, epsilon);
    if (normalization == clamped_sum)
        return weight / fmaxf(denominator, epsilon);
    return weight;
}

} // namespace select
