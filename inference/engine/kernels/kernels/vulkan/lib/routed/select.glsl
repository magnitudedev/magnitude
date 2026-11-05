// Expert ranking at a router's exact SELECT_P width. Instantiated with
// SELECT_FN for current selection and next-layer hints.
// Lane l owns experts l + 32 i; only its `held` entries contribute to
// softmax's reduction. The excluded tail is ranked (-inf, -1). Sorting
// and subgroup argmax use descending score, ties to the higher expert
// index; popping a winner preserves that order for the next rank.
// Scores use the bounded-error exponential from precise.glsl. Selection
// bias affects ranking only; normalization uses unbiased selected scores.
#define SELECT_SOFTMAX 0
#define SELECT_SIGMOID 1
#define SELECT_SQRT_SOFTPLUS 2
#define SELECT_NONE 0
#define SELECT_SUM 1
#define SELECT_SUM_PLUS_EPSILON 2
#define SELECT_CLAMPED_SUM 3

#define SELECT_DEFINE_SCORES \
void SELECT_FN(select_scores)(const int function, inout float values[SELECT_P], const uint held) { \
    if (function == SELECT_SOFTMAX) { \
        float maximum = -uintBitsToFloat(0x7f800000u); \
        [[unroll]] for (uint i = 0u; i < SELECT_P; ++i) \
            if (i < held) \
                maximum = max(maximum, values[i]); \
        maximum = seismic_subgroup_max(maximum); \
        float total = 0.0; \
        [[unroll]] for (uint i = 0u; i < SELECT_P; ++i) { \
            if (i < held) { \
                values[i] = precise_exp(values[i] - maximum); \
                total += values[i]; \
            } \
        } \
        total = seismic_subgroup_sum_f32(total); \
        [[unroll]] for (uint i = 0u; i < SELECT_P; ++i) \
            values[i] = seismic_div_rn(values[i], total); \
    } else if (function == SELECT_SIGMOID) { \
        [[unroll]] for (uint i = 0u; i < SELECT_P; ++i) \
            values[i] = seismic_div_rn(1.0, 1.0 + precise_exp(-values[i])); \
    } else { \
        [[unroll]] for (uint i = 0u; i < SELECT_P; ++i) \
            values[i] = sqrt(max(values[i], 0.0) + log(1.0 + precise_exp(-abs(values[i])))); \
    } \
}

#define SELECT_DEFINE_SORT \
void SELECT_FN(select_sort)(inout float ranked[SELECT_P], inout float score[SELECT_P], inout int expert[SELECT_P]) { \
    [[unroll]] for (uint pass = 0u; pass < SELECT_P; ++pass) { \
        [[unroll]] for (uint i = pass & 1u; i + 1u < SELECT_P; i += 2u) { \
            if (selection_precedes(ranked[i + 1u], expert[i + 1u], ranked[i], expert[i])) { \
                const float r = ranked[i], s = score[i]; \
                const int e = expert[i]; \
                ranked[i] = ranked[i + 1u]; \
                score[i] = score[i + 1u]; \
                expert[i] = expert[i + 1u]; \
                ranked[i + 1u] = r; \
                score[i + 1u] = s; \
                expert[i + 1u] = e; \
            } \
        } \
    } \
}

#define SELECT_DEFINE_POP \
void SELECT_FN(select_pop)(inout float ranked[SELECT_P], inout float score[SELECT_P], inout int expert[SELECT_P]) { \
    [[unroll]] for (uint i = 0u; i + 1u < SELECT_P; ++i) { \
        ranked[i] = ranked[i + 1u]; \
        score[i] = score[i + 1u]; \
        expert[i] = expert[i + 1u]; \
    } \
    ranked[SELECT_P - 1u] = -uintBitsToFloat(0x7f800000u); \
    expert[SELECT_P - 1u] = -1; \
}

#define SELECT_DEFINE_NORMALIZED \
float SELECT_FN(select_normalized)(const int normalization, float weight, float denominator, float epsilon) { \
    if (normalization == SELECT_SUM) \
        return seismic_div_rn(weight, denominator); \
    if (normalization == SELECT_SUM_PLUS_EPSILON) \
        return seismic_div_rn(weight, denominator + epsilon); \
    if (normalization == SELECT_CLAMPED_SUM) \
        return seismic_div_rn(weight, max(denominator, epsilon)); \
    return weight; \
}

#define SELECT_RANKING \
SELECT_DEFINE_SCORES \
SELECT_DEFINE_SORT \
SELECT_DEFINE_POP \
SELECT_DEFINE_NORMALIZED
