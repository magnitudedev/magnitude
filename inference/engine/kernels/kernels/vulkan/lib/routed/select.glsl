// Expert selection of the general routed feed-forward (`routed_select`), one
// subgroup per row: scores, the selection-only bias, top-K and the combine
// weights. The counterpart of `metal/lib/routed/select.h`,
// `cuda/lib/routed/select.cuh` and `cpu/lib/routed/select.rs`.
//
// Lane `lane` holds experts lane + 32 i (i < SELECT_P = E / 32, which the
// including entry defines). Top-K is one pass over a lane-local order: every
// lane sorts its SELECT_P candidates once (an odd-even transposition network
// over registers), then each of the K rounds is a single subgroup argmax over
// the 32 lane heads (ties to the higher expert index) and one shift of the
// winning lane's candidates. The winners are the portable body's K argmax
// rounds in the same order. Scores use the bounded-error `exp`.
//
// This file is independent of any entry ABI.
#include "../core/precise.glsl"

#define SELECT_SOFTMAX 0
#define SELECT_SIGMOID 1
#define SELECT_SQRT_SOFTPLUS 2

#define SELECT_NONE 0
#define SELECT_SUM 1
#define SELECT_SUM_PLUS_EPSILON 2
#define SELECT_CLAMPED_SUM 3

// Descending value, ties to the higher expert index.
bool select_precedes(float value, int expert, float best, int best_expert) {
    return value > best || (value == best && expert > best_expert);
}

// The scores of this lane's experts from their logits (in place).
void select_scores(const int function, inout float values[SELECT_P]) {
    if (function == SELECT_SOFTMAX) {
        float maximum = -uintBitsToFloat(0x7f800000u);
        [[unroll]] for (uint i = 0u; i < SELECT_P; ++i)
            maximum = max(maximum, values[i]);
        maximum = seismic_subgroup_max(maximum);
        float total = 0.0;
        [[unroll]] for (uint i = 0u; i < SELECT_P; ++i) {
            values[i] = precise_exp(values[i] - maximum);
            total += values[i];
        }
        total = seismic_subgroup_sum_f32(total);
        [[unroll]] for (uint i = 0u; i < SELECT_P; ++i)
            values[i] = seismic_div_rn(values[i], total);
    } else if (function == SELECT_SIGMOID) {
        [[unroll]] for (uint i = 0u; i < SELECT_P; ++i)
            values[i] = seismic_div_rn(1.0, 1.0 + precise_exp(-values[i]));
    } else {
        [[unroll]] for (uint i = 0u; i < SELECT_P; ++i)
            values[i] = sqrt(max(values[i], 0.0) + log(1.0 + precise_exp(-abs(values[i]))));
    }
}

// Sorts this lane's (ranked, score, expert) triples by `select_precedes`.
void select_sort(inout float ranked[SELECT_P], inout float score[SELECT_P], inout int expert[SELECT_P]) {
    [[unroll]] for (uint pass = 0u; pass < SELECT_P; ++pass) {
        [[unroll]] for (uint i = pass & 1u; i + 1u < SELECT_P; i += 2u) {
            if (select_precedes(ranked[i + 1u], expert[i + 1u], ranked[i], expert[i])) {
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
void select_pop(inout float ranked[SELECT_P], inout float score[SELECT_P], inout int expert[SELECT_P]) {
    [[unroll]] for (uint i = 0u; i + 1u < SELECT_P; ++i) {
        ranked[i] = ranked[i + 1u];
        score[i] = score[i + 1u];
        expert[i] = expert[i + 1u];
    }
    ranked[SELECT_P - 1u] = -uintBitsToFloat(0x7f800000u);
    expert[SELECT_P - 1u] = -1;
}

// The weight of a selected score after normalization over the slot-order
// sum `denominator`.
float select_normalized(const int normalization, float weight, float denominator, float epsilon) {
    if (normalization == SELECT_SUM)
        return seismic_div_rn(weight, denominator);
    if (normalization == SELECT_SUM_PLUS_EPSILON)
        return seismic_div_rn(weight, denominator + epsilon);
    if (normalization == SELECT_CLAMPED_SUM)
        return seismic_div_rn(weight, max(denominator, epsilon));
    return weight;
}
