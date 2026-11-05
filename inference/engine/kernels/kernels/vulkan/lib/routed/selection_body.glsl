// One numerical recipe, instantiated with each router's exact widths and
// ports. Seismic captures includes once; these function macros provide the
// two compile-time bindings without duplicating arithmetic.
#define SELECT_P ((SELECT_E + 31u) / 32u)
#include "select.glsl"
#define SELECT_TILE 32u
#define SELECT_PITCH 36u

#define SELECT_DEFINE_RESIDUAL \
float SELECT_FN(select_residual)(uint64_t row, uint source) { \
    return element_f32_at(SEISMIC_PTR(SEISMIC_BUFFER_RESIDUAL) \
        + (row * SEISMIC_RESIDUAL_STRIDE_0 + uint64_t(source) * SEISMIC_RESIDUAL_STRIDE_1) * 4ul); \
}

#define SELECT_DEFINE_NORM \
float SELECT_FN(select_norm)(uint source) { \
    return element_at(SELECT_NORM, SEISMIC_PTR(SEISMIC_BUFFER_NORM), uint64_t(source) * SEISMIC_NORM_STRIDE_0); \
}

#define SELECT_DEFINE_ROUTER_NORM \
float SELECT_FN(select_router_norm)(uint source) { \
    return element_at(SELECT_ROUTER_NORM, SELECT_ROUTER_NORM_BUFFER, \
        uint64_t(source) * SELECT_ROUTER_NORM_STRIDE); \
}

#define SELECT_DEFINE_PUBLISH \
void SELECT_FN(select_publish)(uint64_t row, uint source, float value) { \
    element_put(SELECT_ACT, SEISMIC_PTR(SEISMIC_RESULT_0_BUFFER), \
        row * SEISMIC_RESULT_0_STRIDE_0 + uint64_t(source) * SEISMIC_RESULT_0_STRIDE_1, value); \
}

#define SELECT_DEFINE_LOGIT \
uint64_t SELECT_FN(select_logit)(uint64_t row, uint expert) { \
    return SELECT_LOGITS + (row * SELECT_E + expert) * 4ul; \
}

#if !defined(SELECT_DECODE_ONLY)
#define SELECT_DEFINE_ROUTER_ROWS \
uint64_t SELECT_FN(select_router_rows)() { \
    return SELECT_ROUTER_ROWS_BUFFER; \
}
#else
#define SELECT_DEFINE_ROUTER_ROWS
#endif

#if !defined(SEISMIC_ROUTER_KIND_PACKED)
#define SELECT_DEFINE_COLUMN_WEIGHT \
float SELECT_FN(select_column_weight)(uint expert, uint source) { \
    return element_at(packets_element(PACKETS_W0), SELECT_ROUTER_BUFFER, \
        uint64_t(expert) * SELECT_ROUTER_STRIDE_0 + uint64_t(source) * SELECT_ROUTER_STRIDE_1); \
}
#else
#define SELECT_DEFINE_COLUMN_WEIGHT
#endif

#if !defined(SELECT_DECODE_ONLY)
#define SELECT_DEFINE_STAGE \
void SELECT_FN(select_stage)() { \
    const uint64_t row = gl_WorkGroupID.x; \
    const uint thread = gl_LocalInvocationIndex; \
    const uint hidden = uint(SEISMIC_DIM_H); \
    const float eps = element_word_f32(SEISMIC_PARAM_EPSILON); \
    float squares = 0.0; \
    for (uint source = thread; source < hidden; source += SELECT_THREADS) { \
        const float value = SELECT_FN(select_residual)(row, source); \
        squares = seismic_fma_rn(value, value, squares); \
    } \
    const float inverse = inversesqrt(seismic_div_rn(reduce_group_sum(squares, 0u), float(hidden)) + eps); \
    for (uint source = thread; source < hidden; source += SELECT_THREADS) { \
        const float scaled = SELECT_FN(select_residual)(row, source) * inverse; \
        SELECT_FN(select_publish)(row, source, element_round(SELECT_ACT, scaled * SELECT_FN(select_norm)(source))); \
        element_put(SELECT_ACT, SELECT_FN(select_router_rows)(), row * uint64_t(hidden) + source, \
            element_round(SELECT_ACT, scaled * SELECT_FN(select_router_norm)(source))); \
    } \
}
#else
#define SELECT_DEFINE_STAGE
#endif

#if defined(SEISMIC_ROUTER_KIND_PACKED)
#define SELECT_ACCUMULATE_GEMV \
const packets_rows16 router_layout = SELECT_ROUTER_ROWS; \
    const uint64_t router = SELECT_ROUTER_BUFFER; \
    for (uint p = lane; p < hidden / 32u; p += 32u) { \
        const packets_packet packet = packets_load(PACKETS_W0, \
            router + uint64_t(expert) * router_layout.stride, router_layout, p, hidden); \
        [[unroll]] for (uint step = 0u; step < 4u; ++step) { \
            vec4 even, odd; \
            packets_values(PACKETS_W0, packet, step, even, odd); \
            [[unroll]] for (uint i = 0u; i < 8u; ++i) { \
                const uint source = 32u * p + 8u * step + i; \
                const float weight = (i & 1u) != 0u ? odd[i >> 1] : even[i >> 1]; \
                const float scale = SELECT_FN(select_router_norm)(source); \
                [[unroll]] for (uint row = 0u; row < uint(SEISMIC_CONST_DECODE_ROWS); ++row) \
                    if (row < rows) \
                        sums[row] = seismic_fma_rn(element_round(SELECT_ACT, \
                            SELECT_FN(select_residual)(row, source) * seismic_shared_f32[inverses + row] * scale), \
                            weight, sums[row]); \
            } \
        } \
    }
#else
#define SELECT_ACCUMULATE_GEMV \
for (uint base = 4u * lane; base < hidden; base += 128u) { \
        float weight[4], scale[4]; \
        [[unroll]] for (uint j = 0u; j < 4u; ++j) { \
            weight[j] = SELECT_FN(select_column_weight)(expert, base + j); \
            scale[j] = SELECT_FN(select_router_norm)(base + j); \
        } \
        [[unroll]] for (uint row = 0u; row < uint(SEISMIC_CONST_DECODE_ROWS); ++row) { \
            if (row < rows) { \
                const float inverse = seismic_shared_f32[inverses + row]; \
                [[unroll]] for (uint j = 0u; j < 4u; ++j) \
                    sums[row] = seismic_fma_rn(element_round(SELECT_ACT, SELECT_FN(select_residual)(row, base + j) * inverse * scale[j]), \
                        weight[j], sums[row]); \
            } \
        } \
    }
#endif

#define SELECT_DEFINE_GEMV \
void SELECT_FN(select_gemv)() { \
    const uint rows = uint(SEISMIC_DIM_M), hidden = uint(SEISMIC_DIM_H); \
    const uint sg = SEISMIC_SUBGROUP, lane = SEISMIC_LANE, thread = gl_LocalInvocationIndex; \
    const float eps = element_word_f32(SEISMIC_PARAM_EPSILON); \
    const uint inverses = uint(SEISMIC_CONST_DECODE_ROWS) * SELECT_SUBGROUPS; \
    for (uint item = sg; item < rows * SELECT_SUBGROUPS; item += SEISMIC_SUBGROUPS) { \
        const uint row = item / SELECT_SUBGROUPS, part = item % SELECT_SUBGROUPS; \
        float squares = 0.0; \
        for (uint source = part * 32u + lane; source < hidden; source += SELECT_THREADS) { \
            const float value = SELECT_FN(select_residual)(row, source); \
            squares = seismic_fma_rn(value, value, squares); \
        } \
        squares = seismic_subgroup_sum_f32(squares); \
        if (lane == 0u) \
            seismic_shared_f32[row * SELECT_SUBGROUPS + part] = squares; \
    } \
    barrier(); \
    if (thread < rows) { \
        float total = 0.0; \
        for (uint part = 0u; part < SELECT_SUBGROUPS; ++part) \
            total += seismic_shared_f32[thread * SELECT_SUBGROUPS + part]; \
        seismic_shared_f32[inverses + thread] = inversesqrt(seismic_div_rn(total, float(hidden)) + eps); \
    } \
    barrier(); \
    if (SELECT_PUBLISH && gl_WorkGroupID.x == 0u) { \
        for (uint item = thread; item < rows * hidden; item += gl_WorkGroupSize.x) { \
            const uint row = item / hidden, source = item % hidden; \
            SELECT_FN(select_publish)(row, source, \
                element_round(SELECT_ACT, SELECT_FN(select_residual)(row, source) * seismic_shared_f32[inverses + row] * SELECT_FN(select_norm)(source))); \
        } \
    } \
    const uint expert = gl_WorkGroupID.x * SEISMIC_SUBGROUPS + sg; \
    if (expert >= SELECT_E) \
        return; \
    float sums[SEISMIC_CONST_DECODE_ROWS]; \
    [[unroll]] for (uint row = 0u; row < uint(SEISMIC_CONST_DECODE_ROWS); ++row) \
        sums[row] = 0.0; \
SELECT_ACCUMULATE_GEMV \
 \
    [[unroll]] for (uint row = 0u; row < uint(SEISMIC_CONST_DECODE_ROWS); ++row) { \
        if (row < rows) { \
            const float sum = seismic_subgroup_sum_f32(sums[row]); \
            if (lane == 0u) \
                element_f32_put(SELECT_FN(select_logit)(row, expert), sum); \
        } \
    } \
}

#if !defined(SEISMIC_ROUTER_KIND_PACKED)
#define SELECT_STAGE_DENSE_COLUMNS \
const uint expert = expert0 + i; \
            seismic_shared_f32[columns_tile + k * SELECT_PITCH + i] = expert < SELECT_E ? SELECT_FN(select_column_weight)(expert, k0 + k) : 0.0;
#else
#define SELECT_STAGE_DENSE_COLUMNS
#endif

#if defined(SEISMIC_ROUTER_KIND_PACKED)
#define SELECT_STAGE_PACKED_COLUMNS \
if (thread < SELECT_TILE) { \
            const uint expert = expert0 + thread; \
            if (expert < SELECT_E) { \
                const packets_rows16 router_layout = SELECT_ROUTER_ROWS; \
                const uint64_t row = SELECT_ROUTER_BUFFER + uint64_t(expert) * router_layout.stride; \
                const packets_packet packet = packets_load(PACKETS_W0, row, router_layout, k0 / 32u, hidden); \
                [[unroll]] for (uint step = 0u; step < 4u; ++step) { \
                    vec4 even, odd; \
                    packets_values(PACKETS_W0, packet, step, even, odd); \
                    [[unroll]] for (uint i = 0u; i < 8u; ++i) { \
                        seismic_shared_f32[columns_tile + (8u * step + i) * SELECT_PITCH + thread] = \
                            (i & 1u) != 0u ? odd[i >> 1] : even[i >> 1]; \
                    } \
                } \
            } else { \
                for (uint k = 0u; k < SELECT_TILE; ++k) \
                    seismic_shared_f32[columns_tile + k * SELECT_PITCH + thread] = 0.0; \
            } \
        }
#else
#define SELECT_STAGE_PACKED_COLUMNS
#endif

#if !defined(SELECT_DECODE_ONLY)
#define SELECT_DEFINE_GEMM \
void SELECT_FN(select_gemm)() { \
    const uint thread = gl_LocalInvocationIndex; \
    const uint hidden = uint(SEISMIC_DIM_H); \
    const uint expert0 = gl_WorkGroupID.x * SELECT_TILE; \
    const uint64_t row0 = uint64_t(gl_WorkGroupID.y) * SELECT_TILE; \
    const uint columns_tile = SELECT_TILE * SELECT_PITCH; \
    const uint r = thread / 4u, e0 = 8u * (thread % 4u); \
    float acc[8]; \
    [[unroll]] for (uint j = 0u; j < 8u; ++j) \
        acc[j] = 0.0; \
    for (uint k0 = 0u; k0 < hidden; k0 += SELECT_TILE) { \
        barrier(); \
        for (uint item = thread; item < SELECT_TILE * SELECT_TILE; item += gl_WorkGroupSize.x) { \
            const uint i = item / SELECT_TILE, k = item % SELECT_TILE; \
            const uint64_t row = row0 + i; \
            seismic_shared_f32[i * SELECT_PITCH + k] = row < SEISMIC_DIM_M \
                ? element_at(SELECT_ACT, SELECT_FN(select_router_rows)(), row * uint64_t(hidden) + uint64_t(k0 + k)) \
                : 0.0; \
SELECT_STAGE_DENSE_COLUMNS \
 \
        } \
SELECT_STAGE_PACKED_COLUMNS \
 \
        barrier(); \
        for (uint k = 0u; k < SELECT_TILE; ++k) { \
            const float a = seismic_shared_f32[r * SELECT_PITCH + k]; \
            [[unroll]] for (uint j = 0u; j < 8u; ++j) \
                acc[j] = seismic_fma_rn(a, seismic_shared_f32[columns_tile + k * SELECT_PITCH + e0 + j], acc[j]); \
        } \
    } \
    const uint64_t row = row0 + r; \
    [[unroll]] for (uint j = 0u; j < 8u; ++j) \
        if (row < SEISMIC_DIM_M && expert0 + e0 + j < SELECT_E) \
            element_f32_put(SELECT_FN(select_logit)(row, expert0 + e0 + j), acc[j]); \
}
#else
#define SELECT_DEFINE_GEMM
#endif

#define SELECT_DEFINE_RANK \
void SELECT_FN(select_rank)(uint64_t row, out float chosen[SELECT_K], out int winners[SELECT_K]) { \
    const uint lane = SEISMIC_LANE; \
    /* SELECT_P when 32 divides E (a constant: no excluded entries). */ \
    const uint held = selection_candidates(SELECT_E, lane); \
    float score[SELECT_P], ranked[SELECT_P]; \
    int expert[SELECT_P]; \
    [[unroll]] for (uint i = 0u; i < SELECT_P; ++i) { \
        expert[i] = i < held ? int(lane + 32u * i) : -1; \
        score[i] = i < held ? element_f32_at(SELECT_FN(select_logit)(row, lane + 32u * i)) : 0.0; \
    } \
    SELECT_FN(select_scores)(int(SEISMIC_PARAM_SCORE), score, held); \
    [[unroll]] for (uint i = 0u; i < SELECT_P; ++i) \
        ranked[i] = i < held ? selection_ranked(score[i] + element_f32_at(SELECT_BIAS_BUFFER \
            + uint64_t(expert[i]) * SELECT_BIAS_STRIDE * 4ul)) : -uintBitsToFloat(0x7f800000u); \
    SELECT_FN(select_sort)(ranked, score, expert); \
 \
    for (uint rank = 0u; rank < SELECT_K; ++rank) { \
        float best; \
        const int winner = reduce_argmax(ranked[0], expert[0], REDUCE_HIGHER_INDEX, best); \
        /* The winner's unbiased score, from its lane. */ \
        const float winner_score = seismic_shuffle(score[0], uint(winner) % 32u); \
        if (expert[0] == winner) \
            SELECT_FN(select_pop)(ranked, score, expert); \
        chosen[SELECT_K - 1u - rank] = winner_score; \
        winners[SELECT_K - 1u - rank] = winner; \
    } \
}

#define SELECT_DEFINE_SELECT_ROW \
void SELECT_FN(select_select_row)(uint64_t row, out int winners[SELECT_K]) { \
    const uint lane = SEISMIC_LANE; \
    float chosen[SELECT_K]; \
    SELECT_FN(select_rank)(row, chosen, winners); \
    /* The slot-order sum: slot s holds rank K - 1 - s. */ \
    float denominator = 0.0; \
    for (uint slot = 0u; slot < SELECT_K; ++slot) \
        denominator += chosen[slot]; \
    if (lane == 0u) { \
        const int normalization = int(SEISMIC_PARAM_NORMALIZATION); \
        const float scale = element_word_f32(SEISMIC_PARAM_SCALE); \
        const float epsilon = element_word_f32(SEISMIC_PARAM_NORMALIZATION_EPSILON); \
        for (uint slot = 0u; slot < SELECT_K; ++slot) { \
            float weight = SELECT_FN(select_normalized)(normalization, chosen[slot], denominator, epsilon) * scale; \
            if (SEISMIC_DIM_ES != 0ul) \
                weight = weight * element_f32_at(SEISMIC_PTR(SEISMIC_BUFFER_EXPERT_SCALE) \
                    + uint64_t(winners[slot]) * SEISMIC_EXPERT_SCALE_STRIDE_0 * 4ul); \
            element_i32_put(SEISMIC_PTR(SEISMIC_BUFFER_ROUTES) \
                + (row * SEISMIC_ROUTES_STRIDE_0 + uint64_t(slot) * SEISMIC_ROUTES_STRIDE_1) * 4ul, winners[slot]); \
            element_f32_put(SEISMIC_PTR(SEISMIC_BUFFER_WEIGHTS) \
                    + (row * SEISMIC_WEIGHTS_STRIDE_0 + uint64_t(slot) * SEISMIC_WEIGHTS_STRIDE_1) * 4ul, \
                weight); \
        } \
    } \
}

#define SELECT_DEFINE_SELECT \
void SELECT_FN(select_select)() { \
    const uint64_t row = uint64_t(gl_WorkGroupID.x) * SEISMIC_SUBGROUPS + SEISMIC_SUBGROUP; \
    if (row < SEISMIC_DIM_M) { \
        int winners[SELECT_K]; \
        SELECT_FN(select_select_row)(row, winners); \
    } \
}

#define SELECT_NUMERICAL \
SELECT_DEFINE_RESIDUAL \
SELECT_DEFINE_NORM \
SELECT_DEFINE_ROUTER_NORM \
SELECT_DEFINE_PUBLISH \
SELECT_DEFINE_LOGIT \
SELECT_DEFINE_ROUTER_ROWS \
SELECT_DEFINE_COLUMN_WEIGHT \
SELECT_DEFINE_STAGE \
SELECT_DEFINE_GEMV \
SELECT_DEFINE_GEMM \
SELECT_DEFINE_RANK

#define SELECT_OUTPUT \
SELECT_DEFINE_SELECT_ROW \
SELECT_DEFINE_SELECT
