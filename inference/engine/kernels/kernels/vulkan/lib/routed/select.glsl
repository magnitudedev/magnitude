// Expert ranking at a router's exact SELECT_P width. Instantiated with
// SELECT_FN for current selection and next-layer hints.
// Lane l owns experts l + 32 i; only its `held` entries contribute to
// softmax's reduction. Excluded and already-chosen slots hold a quiet NaN
// (`SELECT_EXCLUDED`), which `selection_precedes` never picks. Top-K keeps
// each lane's best remaining candidate `(head_r, head_s, head_e)` seeded by
// one linear scan of its SELECT_P slots, then runs SELECT_K subgroup argmax
// rounds over the 32 lane heads (descending score, ties to the higher expert
// index); on each round before the last, only the winning lane marks its
// winning slot `SELECT_EXCLUDED` and re-scans its SELECT_P slots for the next
// head. Because each slot stays at `lane + 32u * i`, no `expert[SELECT_P]`
// array is stored.
// Scores use the bounded-error exponential from precise.glsl. Selection
// bias affects ranking only; normalization uses unbiased selected scores.
#define SELECT_SOFTMAX 0
#define SELECT_SIGMOID 1
#define SELECT_SQRT_SOFTPLUS 2
#define SELECT_NONE 0
#define SELECT_SUM 1
#define SELECT_SUM_PLUS_EPSILON 2
#define SELECT_CLAMPED_SUM 3
#define SELECT_EXCLUDED uintBitsToFloat(0x7fc00000u)

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

#define SELECT_DEFINE_LANE_MAX \
void SELECT_FN(select_lane_max)(inout float ranked[SELECT_P], const float score[SELECT_P], \
    const uint lane, const int erase, out float head_r, out float head_s, out int head_e) { \
    head_r = -uintBitsToFloat(0x7f800000u); \
    head_s = 0.0; \
    head_e = -1; \
    [[unroll]] for (uint i = 0u; i < SELECT_P; ++i) { \
        const int e = int(lane + 32u * i); \
        if (erase >= 0 && e == erase) \
            ranked[i] = SELECT_EXCLUDED; \
        else if (selection_precedes(ranked[i], e, head_r, head_e)) { \
            head_r = ranked[i]; \
            head_s = score[i]; \
            head_e = e; \
        } \
    } \
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

#define SELECT_DEFINE_RANK_AT \
void SELECT_FN(select_rank_at)(uint64_t logits_base, uint64_t bias_base, uint64_t bias_stride, int score_fn, \
    uint64_t row, out float chosen[SELECT_K], out int winners[SELECT_K]) { \
    const uint lane = SEISMIC_LANE; \
    const uint held = selection_candidates(SELECT_E, lane); \
    float score[SELECT_P], ranked[SELECT_P]; \
    [[unroll]] for (uint i = 0u; i < SELECT_P; ++i) \
        score[i] = i < held \
            ? element_f32_at(logits_base + (row * uint64_t(SELECT_E) + uint64_t(lane + 32u * i)) * 4ul) \
            : 0.0; \
    SELECT_FN(select_scores)(score_fn, score, held); \
    [[unroll]] for (uint i = 0u; i < SELECT_P; ++i) \
        ranked[i] = i < held \
            ? selection_ranked(score[i] + element_f32_at(bias_base + uint64_t(lane + 32u * i) * bias_stride * 4ul)) \
            : SELECT_EXCLUDED; \
    [[unroll]] for (uint i = 0u; i < SELECT_P; ++i) \
        if (i < held) \
            SELECTION_KEY(row, lane + 32u * i, ranked[i]); \
    float head_r, head_s; \
    int head_e; \
    SELECT_FN(select_lane_max)(ranked, score, lane, -1, head_r, head_s, head_e); \
    [[unroll]] for (uint rank = 0u; rank < SELECT_K; ++rank) { \
        float best; \
        const int winner = reduce_argmax(head_r, head_e, REDUCE_HIGHER_INDEX, best); \
        const float winner_score = seismic_shuffle(head_s, uint(winner) % 32u); \
        if (rank + 1u < SELECT_K && head_e == winner) \
            SELECT_FN(select_lane_max)(ranked, score, lane, winner, head_r, head_s, head_e); \
        chosen[SELECT_K - 1u - rank] = winner_score; \
        winners[SELECT_K - 1u - rank] = winner; \
    } \
}

#define SELECT_RANKING \
SELECT_DEFINE_SCORES \
SELECT_DEFINE_LANE_MAX \
SELECT_DEFINE_NORMALIZED \
SELECT_DEFINE_RANK_AT
