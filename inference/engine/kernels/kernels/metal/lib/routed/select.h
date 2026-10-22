// Expert selection of the general routed feed-forward (`routed_select`), one
// simdgroup per row: scores, the selection-only bias, top-K and the combine
// weights. The counterpart of `cuda/lib/routed/select.cuh`,
// `vulkan/lib/routed/select.glsl` and `cpu/lib/routed/select.rs`.
//
// Lane `lane` holds experts lane + 32 i (i < P = ceil(E / 32)); when E is
// not a multiple of 32, lanes at or past E % 32 hold P - 1 (`candidates`)
// and their last register is an excluded entry (ranked -inf, expert -1, as
// a popped one). Top-K is one pass over a lane-local order: every lane sorts
// its P candidates once (an odd-even transposition network over registers),
// then each of the K rounds is a single simdgroup argmax over the 32 lane
// heads (ties to the higher expert index) and one shift of the winning
// lane's candidates. The winners are the portable body's K argmax rounds in
// the same order.
//
// This file is independent of any entry ABI.

#include "../core/reduce.h"

namespace select {

constant constexpr int softmax = 0;
constant constexpr int sigmoid = 1;
constant constexpr int sqrt_softplus = 2;

constant constexpr int no_normalization = 0;
constant constexpr int sum = 1;
constant constexpr int sum_plus_epsilon = 2;
constant constexpr int clamped_sum = 3;

// The experts lane `lane` holds of E: ceil(E / 32) on the first E % 32
// lanes, E / 32 on the others (every lane E / 32 when 32 divides E).
template <uint E>
inline uint candidates(uint lane) {
    return E / 32 + (E % 32 != 0 && lane < E % 32 ? 1u : 0u);
}

// Descending value, ties to the higher expert index.
inline bool precedes(float value, int expert, float best, int best_expert) {
    return value > best || (value == best && expert > best_expert);
}

// The scores of this lane's experts from their logits (in place); the
// softmax runs over the `held` leading values only (the others are excluded
// entries, whose scores are unused). Every loop unrolls fully, so every
// register index is static.
template <uint P>
inline void scores(int function, thread float (&values)[P], uint held) {
    if (function == softmax) {
        float maximum = -INFINITY;
        _Pragma("clang loop unroll(full)")
        for (uint i = 0; i < P; ++i)
            if (i < held)
                maximum = metal::max(maximum, values[i]);
        maximum = simd_max(maximum);
        float total = 0.0f;
        _Pragma("clang loop unroll(full)")
        for (uint i = 0; i < P; ++i) {
            if (i < held) {
                values[i] = metal::exp(values[i] - maximum);
                total += values[i];
            }
        }
        total = simd_sum(total);
        _Pragma("clang loop unroll(full)")
        for (uint i = 0; i < P; ++i)
            values[i] = values[i] / total;
    } else if (function == sigmoid) {
        _Pragma("clang loop unroll(full)")
        for (uint i = 0; i < P; ++i)
            values[i] = 1.0f / (1.0f + metal::exp(-values[i]));
    } else {
        _Pragma("clang loop unroll(full)")
        for (uint i = 0; i < P; ++i)
            values[i] = metal::sqrt(metal::max(values[i], 0.0f) + metal::log(1.0f + metal::exp(-metal::abs(values[i]))));
    }
}

// Sorts this lane's (ranked, score, expert) triples by `precedes`. Both loops
// unroll fully, so every register index is static.
template <uint P>
inline void sort(thread float (&ranked)[P], thread float (&score)[P], thread int (&expert)[P]) {
    _Pragma("clang loop unroll(full)")
    for (uint pass = 0; pass < P; ++pass) {
        _Pragma("clang loop unroll(full)")
        for (uint i = pass & 1u; i + 1u < P; i += 2u) {
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

// The weight of a selected score after normalization over the slot-order
// sum `denominator`.
inline float normalized(int normalization, float weight, float denominator, float epsilon) {
    if (normalization == sum)
        return weight / denominator;
    if (normalization == sum_plus_epsilon)
        return weight / (denominator + epsilon);
    if (normalization == clamped_sum)
        return weight / metal::max(denominator, epsilon);
    return weight;
}

} // namespace select
