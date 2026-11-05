// One numerical recipe, instantiated with each router's exact widths and
// ports. Seismic captures includes once; these function macros provide the
// two compile-time bindings without duplicating arithmetic.
#define ROUTE_PER_LANE ((ROUTE_E + 31u) / 32u)
#define ROUTE_TILE 32u
#define ROUTE_PITCH 36u

#define ROUTE_DEFINE_RESIDUAL \
float ROUTE_FN(route_residual)(uint64_t row, uint source) { \
    return element_f32_at(SEISMIC_PTR(SEISMIC_BUFFER_RESIDUAL) \
        + (row * SEISMIC_RESIDUAL_STRIDE_0 + uint64_t(source) * SEISMIC_RESIDUAL_STRIDE_1) * 4ul); \
}

#if SEISMIC_STATIC_NM == 0
#define ROUTE_READ_NORM \
return 0.0;
#else
#define ROUTE_READ_NORM \
return element_at(ROUTE_NORM, ROUTE_NORM_BUFFER, uint64_t(source) * ROUTE_NORM_STRIDE);
#endif

#define ROUTE_DEFINE_NORM \
float ROUTE_FN(route_norm)(uint source) { \
ROUTE_READ_NORM \
 \
}

#if SEISMIC_STATIC_NM == 0
#define ROUTE_FORM_INPUT \
float sum = 0.0; \
    for (uint part = 0u; part < uint(SEISMIC_DIM_PT); ++part) \
        sum = seismic_add_rn(sum, ROUTE_FN(route_residual)(row, part * uint(SEISMIC_DIM_H) + source)); \
    return element_round(ROUTE_ACT, sum);
#else
#define ROUTE_FORM_INPUT \
return element_round(ROUTE_ACT, ROUTE_FN(route_residual)(row, source) * inverse * scale);
#endif

#define ROUTE_DEFINE_INPUT \
float ROUTE_FN(route_input)(uint64_t row, uint source, float inverse, float scale) { \
ROUTE_FORM_INPUT \
 \
}

#define ROUTE_DEFINE_PUBLISH \
void ROUTE_FN(route_publish)(uint64_t row, uint source, float value) { \
    element_put(ROUTE_ACT, SEISMIC_PTR(SEISMIC_RESULT_0_BUFFER), \
        row * SEISMIC_RESULT_0_STRIDE_0 + uint64_t(source) * SEISMIC_RESULT_0_STRIDE_1, value); \
}

#define ROUTE_DEFINE_LOGIT \
uint64_t ROUTE_FN(route_logit)(uint64_t row, uint expert) { \
    return ROUTE_LOGITS + (row * (ROUTE_E + 1u) + expert) * 4ul; \
}

#if !defined(SEISMIC_ROUTER_KIND_PACKED)
#define ROUTE_DEFINE_COLUMN_WEIGHT \
float ROUTE_FN(route_column_weight)(uint expert, uint source) { \
    return expert < ROUTE_E \
        ? element_at(packets_element(PACKETS_W0), ROUTE_ROUTER_BUFFER, \
            uint64_t(expert) * ROUTE_ROUTER_STRIDE_0 + uint64_t(source) * ROUTE_ROUTER_STRIDE_1) \
        : element_f32_at(SEISMIC_PTR(SEISMIC_BUFFER_SHARED_ROUTER) + uint64_t(source) * SEISMIC_SHARED_ROUTER_STRIDE_0 * 4ul); \
}
#else
#define ROUTE_DEFINE_COLUMN_WEIGHT
#endif

#if !defined(ROUTE_DECODE_ONLY)
#define ROUTE_DEFINE_STAGE \
void ROUTE_FN(route_stage)() { \
    const uint64_t row = gl_WorkGroupID.x; \
    const uint thread = gl_LocalInvocationIndex; \
    const uint hidden = uint(SEISMIC_DIM_H); \
    const float eps = element_word_f32(SEISMIC_PARAM_EPS); \
    float squares = 0.0; \
    for (uint source = thread; source < hidden; source += ROUTE_THREADS) { \
        const float value = ROUTE_FN(route_residual)(row, source); \
        squares = seismic_fma_rn(value, value, squares); \
    } \
    const float inverse = inversesqrt(seismic_div_rn(reduce_group_sum(squares, 0u), float(hidden)) + eps); \
    for (uint source = thread; source < hidden; source += ROUTE_THREADS) \
        ROUTE_FN(route_publish)(row, source, ROUTE_FN(route_input)(row, source, inverse, ROUTE_FN(route_norm)(source))); \
}
#else
#define ROUTE_DEFINE_STAGE
#endif

#if defined(SEISMIC_ROUTER_KIND_PACKED)
#define ROUTE_ACCUMULATE_GEMV \
const packets_rows16 router_layout = ROUTE_ROUTER_ROWS; \
    const uint64_t router = ROUTE_ROUTER_BUFFER; \
    for (uint p = lane; p < hidden / 32u; p += 32u) { \
        if (expert < ROUTE_E) { \
            const packets_packet packet = packets_load(PACKETS_W0, \
                router + uint64_t(expert) * router_layout.stride, router_layout, p, hidden); \
            [[unroll]] for (uint step = 0u; step < 4u; ++step) { \
                vec4 even, odd; \
                packets_values(PACKETS_W0, packet, step, even, odd); \
                [[unroll]] for (uint i = 0u; i < 8u; ++i) { \
                    const uint source = 32u * p + 8u * step + i; \
                    const float weight = (i & 1u) != 0u ? odd[i >> 1] : even[i >> 1]; \
                    const float scale = ROUTE_FN(route_norm)(source); \
                    [[unroll]] for (uint row = 0u; row < uint(SEISMIC_CONST_DECODE_ROWS); ++row) \
                        if (row < rows) \
                            sums[row] = seismic_fma_rn( \
                                ROUTE_FN(route_input)(row, source, seismic_shared_f32[inverses + row], scale), weight, sums[row]); \
                } \
            } \
        } else { \
            for (uint i = 0u; i < 32u; ++i) { \
                const uint source = 32u * p + i; \
                const float weight = element_f32_at(SEISMIC_PTR(SEISMIC_BUFFER_SHARED_ROUTER) \
                    + uint64_t(source) * SEISMIC_SHARED_ROUTER_STRIDE_0 * 4ul); \
                const float scale = ROUTE_FN(route_norm)(source); \
                [[unroll]] for (uint row = 0u; row < uint(SEISMIC_CONST_DECODE_ROWS); ++row) \
                    if (row < rows) \
                        sums[row] = seismic_fma_rn( \
                            ROUTE_FN(route_input)(row, source, seismic_shared_f32[inverses + row], scale), weight, sums[row]); \
            } \
        } \
    }
#else
#define ROUTE_ACCUMULATE_GEMV \
for (uint base = 4u * lane; base < hidden; base += 128u) { \
        float weight[4], scale[4]; \
        [[unroll]] for (uint j = 0u; j < 4u; ++j) { \
            weight[j] = ROUTE_FN(route_column_weight)(expert, base + j); \
            scale[j] = ROUTE_FN(route_norm)(base + j); \
        } \
        [[unroll]] for (uint row = 0u; row < uint(SEISMIC_CONST_DECODE_ROWS); ++row) { \
            if (row < rows) { \
                const float inverse = seismic_shared_f32[inverses + row]; \
                [[unroll]] for (uint j = 0u; j < 4u; ++j) \
                    sums[row] = seismic_fma_rn(ROUTE_FN(route_input)(row, base + j, inverse, scale[j]), weight[j], sums[row]); \
            } \
        } \
    }
#endif

#define ROUTE_DEFINE_GEMV \
void ROUTE_FN(route_gemv)() { \
    const uint rows = uint(SEISMIC_DIM_M), hidden = uint(SEISMIC_DIM_H); \
    const uint sg = SEISMIC_SUBGROUP, lane = SEISMIC_LANE, thread = gl_LocalInvocationIndex; \
    const float eps = element_word_f32(SEISMIC_PARAM_EPS); \
    const uint inverses = uint(SEISMIC_CONST_DECODE_ROWS) * ROUTE_SUBGROUPS; \
    /* The stage launch's sum order (256 strided partials, subgroup sums, then */ \
    /* the eight subgroup partials in order): item (row, part) is part `part`'s */ \
    /* subgroup sum, and the items spread over every subgroup. */ \
    for (uint item = sg; item < rows * ROUTE_SUBGROUPS; item += SEISMIC_SUBGROUPS) { \
        const uint row = item / ROUTE_SUBGROUPS, part = item % ROUTE_SUBGROUPS; \
        float squares = 0.0; \
        for (uint source = part * 32u + lane; source < hidden; source += ROUTE_THREADS) { \
            const float value = ROUTE_FN(route_residual)(row, source); \
            squares = seismic_fma_rn(value, value, squares); \
        } \
        squares = seismic_subgroup_sum_f32(squares); \
        if (lane == 0u) \
            seismic_shared_f32[row * ROUTE_SUBGROUPS + part] = squares; \
    } \
    barrier(); \
    if (thread < rows) { \
        float total = 0.0; \
        for (uint part = 0u; part < ROUTE_SUBGROUPS; ++part) \
            total += seismic_shared_f32[thread * ROUTE_SUBGROUPS + part]; \
        seismic_shared_f32[inverses + thread] = inversesqrt(seismic_div_rn(total, float(hidden)) + eps); \
    } \
    barrier(); \
    if (ROUTE_PUBLISH && gl_WorkGroupID.x == 0u) { \
        for (uint item = thread; item < rows * hidden; item += gl_WorkGroupSize.x) { \
            const uint row = item / hidden, source = item % hidden; \
            ROUTE_FN(route_publish)(row, source, ROUTE_FN(route_input)(row, source, seismic_shared_f32[inverses + row], ROUTE_FN(route_norm)(source))); \
        } \
    } \
    const uint expert = gl_WorkGroupID.x * SEISMIC_SUBGROUPS + sg; \
    if (expert >= ROUTE_E + uint(ROUTE_PUBLISH)) \
        return; \
    /* Lane l owns sources 4l + 128 i .. + 3 (four independent loads per */ \
    /* step), accumulated in source order within the lane. */ \
    float sums[SEISMIC_CONST_DECODE_ROWS]; \
    [[unroll]] for (uint row = 0u; row < uint(SEISMIC_CONST_DECODE_ROWS); ++row) \
        sums[row] = 0.0; \
ROUTE_ACCUMULATE_GEMV \
 \
    [[unroll]] for (uint row = 0u; row < uint(SEISMIC_CONST_DECODE_ROWS); ++row) { \
        if (row < rows) { \
            const float sum = seismic_subgroup_sum_f32(sums[row]); \
            if (lane == 0u) \
                element_f32_put(ROUTE_FN(route_logit)(row, expert), sum); \
        } \
    } \
}

#if !defined(SEISMIC_ROUTER_KIND_PACKED)
#define ROUTE_STAGE_DENSE_COLUMNS \
seismic_shared_f32[columns_tile + k * ROUTE_PITCH + i] = expert <= ROUTE_E ? ROUTE_FN(route_column_weight)(expert, k0 + k) : 0.0;
#else
#define ROUTE_STAGE_DENSE_COLUMNS
#endif

#if defined(SEISMIC_ROUTER_KIND_PACKED)
#define ROUTE_STAGE_PACKED_COLUMNS \
if (thread < ROUTE_TILE) { \
            const uint expert = expert0 + thread; \
            if (expert < ROUTE_E) { \
                const packets_rows16 router_layout = ROUTE_ROUTER_ROWS; \
                const uint64_t row = ROUTE_ROUTER_BUFFER + uint64_t(expert) * router_layout.stride; \
                const packets_packet packet = packets_load(PACKETS_W0, row, router_layout, k0 / 32u, hidden); \
                [[unroll]] for (uint step = 0u; step < 4u; ++step) { \
                    vec4 even, odd; \
                    packets_values(PACKETS_W0, packet, step, even, odd); \
                    [[unroll]] for (uint i = 0u; i < 8u; ++i) { \
                        seismic_shared_f32[columns_tile + (8u * step + i) * ROUTE_PITCH + thread] = \
                            (i & 1u) != 0u ? odd[i >> 1] : even[i >> 1]; \
                    } \
                } \
            } else { \
                for (uint k = 0u; k < ROUTE_TILE; ++k) \
                    seismic_shared_f32[columns_tile + k * ROUTE_PITCH + thread] = expert == ROUTE_E \
                        ? element_f32_at(SEISMIC_PTR(SEISMIC_BUFFER_SHARED_ROUTER) \
                            + uint64_t(k0 + k) * SEISMIC_SHARED_ROUTER_STRIDE_0 * 4ul) : 0.0; \
            } \
        }
#else
#define ROUTE_STAGE_PACKED_COLUMNS
#endif

#if !defined(ROUTE_DECODE_ONLY)
#define ROUTE_DEFINE_GEMM \
void ROUTE_FN(route_gemm)() { \
    const uint thread = gl_LocalInvocationIndex; \
    const uint hidden = uint(SEISMIC_DIM_H); \
    const uint expert0 = gl_WorkGroupID.x * ROUTE_TILE; \
    const uint64_t row0 = uint64_t(gl_WorkGroupID.y) * ROUTE_TILE; \
    const uint columns_tile = ROUTE_TILE * ROUTE_PITCH; \
    const uint r = thread / 4u, e0 = 8u * (thread % 4u); \
    float acc[8]; \
    [[unroll]] for (uint j = 0u; j < 8u; ++j) \
        acc[j] = 0.0; \
    for (uint k0 = 0u; k0 < hidden; k0 += ROUTE_TILE) { \
        barrier(); \
        for (uint item = thread; item < ROUTE_TILE * ROUTE_TILE; item += gl_WorkGroupSize.x) { \
            const uint i = item / ROUTE_TILE, k = item % ROUTE_TILE; \
            const uint64_t row = row0 + i; \
            const uint expert = expert0 + i; \
            seismic_shared_f32[i * ROUTE_PITCH + k] = row < SEISMIC_DIM_M \
                ? element_at(ROUTE_ACT, SEISMIC_PTR(SEISMIC_RESULT_0_BUFFER), \
                    row * SEISMIC_RESULT_0_STRIDE_0 + uint64_t(k0 + k) * SEISMIC_RESULT_0_STRIDE_1) \
                : 0.0; \
ROUTE_STAGE_DENSE_COLUMNS \
 \
        } \
ROUTE_STAGE_PACKED_COLUMNS \
 \
        barrier(); \
        for (uint k = 0u; k < ROUTE_TILE; ++k) { \
            const float a = seismic_shared_f32[r * ROUTE_PITCH + k]; \
            [[unroll]] for (uint j = 0u; j < 8u; ++j) \
                acc[j] = seismic_fma_rn(a, seismic_shared_f32[columns_tile + k * ROUTE_PITCH + e0 + j], acc[j]); \
        } \
    } \
    const uint64_t row = row0 + r; \
    [[unroll]] for (uint j = 0u; j < 8u; ++j) \
        if (row < SEISMIC_DIM_M && expert0 + e0 + j <= ROUTE_E) \
            element_f32_put(ROUTE_FN(route_logit)(row, expert0 + e0 + j), acc[j]); \
}
#else
#define ROUTE_DEFINE_GEMM
#endif

#define ROUTE_DEFINE_RANK \
void ROUTE_FN(route_rank)(uint64_t row, out float chosen[ROUTE_K], out int winners[ROUTE_K]) { \
    const uint lane = SEISMIC_LANE; \
    /* ROUTE_PER_LANE when 32 divides E (a constant). */ \
    const uint held = selection_candidates(ROUTE_E, lane); \
    float probability[ROUTE_PER_LANE]; \
    int expert[ROUTE_PER_LANE]; \
    float maximum = -uintBitsToFloat(0x7f800000u); \
    [[unroll]] for (uint i = 0u; i < ROUTE_PER_LANE; ++i) { \
        expert[i] = i < held ? int(lane + 32u * i) : -1; \
        probability[i] = i < held ? element_f32_at(ROUTE_FN(route_logit)(row, lane + 32u * i)) : -uintBitsToFloat(0x7f800000u); \
        maximum = max(maximum, probability[i]); \
    } \
    maximum = seismic_subgroup_max(maximum); \
    float total = 0.0; \
    [[unroll]] for (uint i = 0u; i < ROUTE_PER_LANE; ++i) { \
        if (i < held) { \
            probability[i] = exp(probability[i] - maximum); \
            total += probability[i]; \
        } \
    } \
    total = seismic_subgroup_sum_f32(total); \
    [[unroll]] for (uint i = 0u; i < ROUTE_PER_LANE; ++i) \
        if (i < held) \
            probability[i] = selection_ranked(seismic_div_rn(probability[i], total)); \
 \
    for (uint rank = 0u; rank < ROUTE_K; ++rank) { \
        float best = -uintBitsToFloat(0x7f800000u); \
        int best_expert = -1; \
        [[unroll]] for (uint i = 0u; i < ROUTE_PER_LANE; ++i) { \
            if (selection_precedes(probability[i], expert[i], best, best_expert)) { \
                best = probability[i]; \
                best_expert = expert[i]; \
            } \
        } \
        float winner_probability; \
        const int winner = reduce_argmax(best, best_expert, REDUCE_HIGHER_INDEX, winner_probability); \
        [[unroll]] for (uint i = 0u; i < ROUTE_PER_LANE; ++i) { \
            if (winner == expert[i]) { \
                probability[i] = -uintBitsToFloat(0x7f800000u); \
                expert[i] = -1; \
            } \
        } \
        chosen[rank] = winner_probability; \
        winners[rank] = winner; \
    } \
}

#define ROUTE_DEFINE_SELECT_ROW \
void ROUTE_FN(route_select_row)(uint64_t row, out int winners[ROUTE_K]) { \
    const uint lane = SEISMIC_LANE; \
    const uint64_t routes = SEISMIC_PTR(SEISMIC_BUFFER_ROUTES); \
    const uint64_t scores = SEISMIC_PTR(SEISMIC_BUFFER_SCORES); \
    float chosen[ROUTE_K]; \
    ROUTE_FN(route_rank)(row, chosen, winners); \
    if (lane == 0u) \
        for (uint slot = 0u; slot < ROUTE_K; ++slot) \
            element_i32_put(routes + (row * SEISMIC_ROUTES_STRIDE_0 + uint64_t(slot) * SEISMIC_ROUTES_STRIDE_1) * 4ul, winners[ROUTE_K - 1u - slot]); \
    /* Slot s holds rank K - 1 - s; the renormalization divides by the */ \
    /* slot-order sum, then each score is multiplied by its expert's scale. */ \
    float denominator = 0.0; \
    if (SEISMIC_PARAM_NORMALIZE != 0ul) \
        for (uint slot = 0u; slot < ROUTE_K; ++slot) \
            denominator += chosen[ROUTE_K - 1u - slot]; \
    if (lane == 0u) { \
        for (uint slot = 0u; slot < ROUTE_K; ++slot) { \
            const uint rank = ROUTE_K - 1u - slot; \
            float score = SEISMIC_PARAM_NORMALIZE != 0ul ? seismic_div_rn(chosen[rank], denominator) : chosen[rank]; \
            if (SEISMIC_DIM_ES != 0ul) \
                score = seismic_mul_rn(score, element_f32_at(SEISMIC_PTR(SEISMIC_BUFFER_EXPERT_SCALE) \
                    + uint64_t(winners[rank]) * SEISMIC_EXPERT_SCALE_STRIDE_0 * 4ul)); \
            element_f32_put(scores + (row * SEISMIC_SCORES_STRIDE_0 + uint64_t(slot) * SEISMIC_SCORES_STRIDE_1) * 4ul, \
                score); \
        } \
    } \
    if (lane == 0u) \
        element_f32_put(SEISMIC_PTR(SEISMIC_RESULT_1_BUFFER) + row * SEISMIC_RESULT_1_STRIDE_0 * 4ul, \
            seismic_div_rn(1.0, 1.0 + exp(-element_f32_at(ROUTE_FN(route_logit)(row, ROUTE_E))))); \
}

#define ROUTE_DEFINE_SELECT \
void ROUTE_FN(route_select)() { \
    const uint64_t row = uint64_t(gl_WorkGroupID.x) * SEISMIC_SUBGROUPS + SEISMIC_SUBGROUP; \
    if (row < SEISMIC_DIM_M) { \
        int winners[ROUTE_K]; \
        ROUTE_FN(route_select_row)(row, winners); \
    } \
}

#define ROUTE_NUMERICAL \
ROUTE_DEFINE_RESIDUAL \
ROUTE_DEFINE_NORM \
ROUTE_DEFINE_INPUT \
ROUTE_DEFINE_PUBLISH \
ROUTE_DEFINE_LOGIT \
ROUTE_DEFINE_COLUMN_WEIGHT \
ROUTE_DEFINE_STAGE \
ROUTE_DEFINE_GEMV \
ROUTE_DEFINE_GEMM \
ROUTE_DEFINE_RANK

#define ROUTE_OUTPUT \
ROUTE_DEFINE_SELECT_ROW \
ROUTE_DEFINE_SELECT
