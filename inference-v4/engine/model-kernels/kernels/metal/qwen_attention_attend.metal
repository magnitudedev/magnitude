inline ulong at2(ulong a, ulong b, ulong sa, ulong sb) { return a * sa + b * sb; }
inline ulong at3(ulong a, ulong b, ulong c, ulong sa, ulong sb, ulong sc) {
    return a * sa + b * sb + c * sc;
}
inline float activation_load(device const uchar *base, ulong logical) {
#if defined(SEISMIC_ELEMENT_A_REPRESENTATION_F32)
    return *reinterpret_cast<device const float *>(base + logical * 4);
#elif defined(SEISMIC_ELEMENT_A_REPRESENTATION_F16)
    return float(*reinterpret_cast<device const half *>(base + logical * 2));
#elif defined(SEISMIC_ELEMENT_A_REPRESENTATION_BF16)
    return as_type<float>(uint(*reinterpret_cast<device const ushort *>(base + logical * 2)) << 16);
#else
#error "attention activation must be dense"
#endif
}
inline void activation_store(device uchar *base, ulong logical, float value) {
#if defined(SEISMIC_ELEMENT_A_REPRESENTATION_F32)
    *reinterpret_cast<device float *>(base + logical * 4) = value;
#elif defined(SEISMIC_ELEMENT_A_REPRESENTATION_F16)
    *reinterpret_cast<device half *>(base + logical * 2) = half(value);
#elif defined(SEISMIC_ELEMENT_A_REPRESENTATION_BF16)
    uint bits = as_type<uint>(value); bits += 0x7fffu + ((bits >> 16) & 1u);
    *reinterpret_cast<device ushort *>(base + logical * 2) = ushort(bits >> 16);
#endif
}
inline float stage_query(device const uchar *query, ulong row, ulong head, ulong column,
    constant ulong *seismic_words) {
    return activation_load(query, at3(row, head, column, SEISMIC_QUERY_STRIDE_0,
        SEISMIC_QUERY_STRIDE_1, SEISMIC_QUERY_STRIDE_2));
}
inline float stage_key(device const uchar *key, ulong row, ulong head, ulong column,
    constant ulong *seismic_words) {
    return activation_load(key, at3(row, head, column, SEISMIC_PREPARED_KEY_STRIDE_0,
        SEISMIC_PREPARED_KEY_STRIDE_1, SEISMIC_PREPARED_KEY_STRIDE_2));
}
inline float stage_value(device const uchar *value, ulong row, ulong head, ulong column,
    constant ulong *seismic_words) {
    return activation_load(value, row * SEISMIC_VALUE_STRIDE_0
        + (head * SEISMIC_DIM_W + column) * SEISMIC_VALUE_STRIDE_1);
}
inline float stage_score(device const uchar *query, device const uchar *key,
    ulong row, ulong query_head, ulong token, ulong kv_head, bool historical,
    constant ulong *seismic_words) {
    float score = 0.0f;
    for (ulong column = 0; column < SEISMIC_DIM_W; ++column) {
        float q = stage_query(query, row, query_head, column, seismic_words);
        float k = historical
            ? activation_load(key, at3(token, kv_head, column, SEISMIC_HISTORY_KEY_STRIDE_0,
                SEISMIC_HISTORY_KEY_STRIDE_1, SEISMIC_HISTORY_KEY_STRIDE_2))
            : stage_key(key, token, kv_head, column, seismic_words);
        score = metal::fma(q, k, score);
    }
    return score * as_type<float>(uint(SEISMIC_PARAM_SCALE));
}
kernel void qwen_attention_attend(
    device const uchar *query [[buffer(SEISMIC_BUFFER_QUERY)]],
    device const uchar *prepared_key [[buffer(SEISMIC_BUFFER_PREPARED_KEY)]],
    device const uchar *value [[buffer(SEISMIC_BUFFER_VALUE)]],
    device const uchar *gate [[buffer(SEISMIC_BUFFER_GATE)]],
    device const int *visible [[buffer(SEISMIC_BUFFER_VISIBLE)]],
    device const int *fresh [[buffer(SEISMIC_BUFFER_FRESH)]],
    device const uchar *history_key [[buffer(SEISMIC_BUFFER_HISTORY_KEY)]],
    device const uchar *history_value [[buffer(SEISMIC_BUFFER_HISTORY_VALUE)]],
    device float *accumulator [[buffer(SEISMIC_BUFFER_ACCUMULATOR)]],
    device uchar *gated [[buffer(SEISMIC_RESULT_0_BUFFER)]],
    constant ulong *seismic_words [[buffer(SEISMIC_BUFFER_WORDS)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint thread_in_group [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd_width [[threads_per_simdgroup]]) {
    ulong query_heads = SEISMIC_DIM_KV * SEISMIC_DIM_G;
    ulong query_head = ulong(group.x);
    uint subgroup = thread_in_group / simd_width;
    ulong row = ulong(group.y) * 8 + ulong(subgroup);
    ulong kv_head = query_head / SEISMIC_DIM_G;
    // Threadgroup storage has a compile-time bound. Wider heads retain the
    // ordinary per-head path; both paths use the same checked tensor contract.
    if (SEISMIC_DIM_M == 1 && SEISMIC_DIM_W <= 256) {
        // At decode there is one query row and only one group per head. Split
        // its visible and fresh keys across all eight SIMD groups. Each group
        // owns an independent online-softmax state; the states merge after a
        // single threadgroup barrier. No numerical accumulator is staged in
        // device memory for every history token.
        threadgroup float partial[8 * 256];
        threadgroup float maxima[8];
        threadgroup float denominators[8];
        for (ulong column = ulong(lane); column < SEISMIC_DIM_W;
             column += ulong(simd_width))
            partial[ulong(subgroup) * SEISMIC_DIM_W + column] = 0.0f;
        float maximum = -INFINITY;
        float denominator = 0.0f;
        float scale = as_type<float>(uint(SEISMIC_PARAM_SCALE));
        for (ulong span = 0; span < SEISMIC_DIM_R + 1; ++span) {
            int lo = span < SEISMIC_DIM_R
                ? visible[at3(0, span, 0, SEISMIC_VISIBLE_STRIDE_0,
                    SEISMIC_VISIBLE_STRIDE_1, SEISMIC_VISIBLE_STRIDE_2)]
                : fresh[at2(0, 0, SEISMIC_FRESH_STRIDE_0, SEISMIC_FRESH_STRIDE_1)];
            int hi = span < SEISMIC_DIM_R
                ? visible[at3(0, span, 1, SEISMIC_VISIBLE_STRIDE_0,
                    SEISMIC_VISIBLE_STRIDE_1, SEISMIC_VISIBLE_STRIDE_2)]
                : fresh[at2(0, 1, SEISMIC_FRESH_STRIDE_0, SEISMIC_FRESH_STRIDE_1)];
            bool historical = span < SEISMIC_DIM_R;
            for (int token = lo + int(subgroup); token < hi; token += 8) {
                float score_part = 0.0f;
                for (ulong column = ulong(lane); column < SEISMIC_DIM_W;
                     column += ulong(simd_width)) {
                    float q = stage_query(query, 0, query_head, column, seismic_words);
                    float k = historical
                        ? activation_load(history_key, at3(ulong(token), kv_head, column,
                            SEISMIC_HISTORY_KEY_STRIDE_0, SEISMIC_HISTORY_KEY_STRIDE_1,
                            SEISMIC_HISTORY_KEY_STRIDE_2))
                        : stage_key(prepared_key, ulong(token), kv_head, column, seismic_words);
                    score_part = metal::fma(q, k, score_part);
                }
                float score = simd_sum(score_part) * scale;
                float next_maximum = metal::max(maximum, score);
                float carry = metal::exp(maximum - next_maximum);
                float probability = metal::exp(score - next_maximum);
                denominator = denominator * carry + probability;
                for (ulong column = ulong(lane); column < SEISMIC_DIM_W;
                     column += ulong(simd_width)) {
                    float v = historical
                        ? activation_load(history_value, at3(ulong(token), kv_head, column,
                            SEISMIC_HISTORY_VALUE_STRIDE_0, SEISMIC_HISTORY_VALUE_STRIDE_1,
                            SEISMIC_HISTORY_VALUE_STRIDE_2))
                        : stage_value(value, ulong(token), kv_head, column, seismic_words);
                    ulong at = ulong(subgroup) * SEISMIC_DIM_W + column;
                    partial[at] = metal::fma(carry, partial[at], probability * v);
                }
                maximum = next_maximum;
            }
        }
        if (lane == 0) {
            maxima[subgroup] = maximum;
            denominators[subgroup] = denominator;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (subgroup == 0) {
            float total_maximum = -INFINITY;
            for (uint part = 0; part < 8; ++part)
                if (denominators[part] > 0.0f)
                    total_maximum = metal::max(total_maximum, maxima[part]);
            float total_denominator = 0.0f;
            for (uint part = 0; part < 8; ++part)
                if (denominators[part] > 0.0f)
                    total_denominator += denominators[part]
                        * metal::exp(maxima[part] - total_maximum);
            for (ulong column = ulong(lane); column < SEISMIC_DIM_W;
                 column += ulong(simd_width)) {
                float weighted = 0.0f;
                for (uint part = 0; part < 8; ++part)
                    if (denominators[part] > 0.0f)
                        weighted = metal::fma(
                            metal::exp(maxima[part] - total_maximum),
                            partial[ulong(part) * SEISMIC_DIM_W + column], weighted);
                float attended = total_denominator > 0.0f
                    ? weighted / total_denominator : 0.0f;
                float gate_value = activation_load(gate, at3(0, query_head, column,
                    SEISMIC_GATE_STRIDE_0, SEISMIC_GATE_STRIDE_1, SEISMIC_GATE_STRIDE_2));
                activation_store(gated, at3(0, query_head, column,
                    SEISMIC_RESULT_0_STRIDE_0, SEISMIC_RESULT_0_STRIDE_1,
                    SEISMIC_RESULT_0_STRIDE_2),
                    attended / (1.0f + metal::exp(-gate_value)));
            }
        }
        return;
    }
    if (row >= SEISMIC_DIM_M) return;
    for (ulong column = ulong(lane); column < SEISMIC_DIM_W; column += ulong(simd_width))
        accumulator[at3(row, query_head, column, SEISMIC_ACCUMULATOR_STRIDE_0,
            SEISMIC_ACCUMULATOR_STRIDE_1, SEISMIC_ACCUMULATOR_STRIDE_2)] = 0.0f;
    float maximum = -INFINITY;
    float denominator = 0.0f;
    float scale = as_type<float>(uint(SEISMIC_PARAM_SCALE));
    for (ulong span = 0; span < SEISMIC_DIM_R + 1; ++span) {
        int lo = span < SEISMIC_DIM_R
            ? visible[at3(row, span, 0, SEISMIC_VISIBLE_STRIDE_0,
                SEISMIC_VISIBLE_STRIDE_1, SEISMIC_VISIBLE_STRIDE_2)]
            : fresh[at2(row, 0, SEISMIC_FRESH_STRIDE_0, SEISMIC_FRESH_STRIDE_1)];
        int hi = span < SEISMIC_DIM_R
            ? visible[at3(row, span, 1, SEISMIC_VISIBLE_STRIDE_0,
                SEISMIC_VISIBLE_STRIDE_1, SEISMIC_VISIBLE_STRIDE_2)]
            : fresh[at2(row, 1, SEISMIC_FRESH_STRIDE_0, SEISMIC_FRESH_STRIDE_1)];
        bool historical = span < SEISMIC_DIM_R;
        for (int token = lo; token < hi; ++token) {
            float partial = 0.0f;
            for (ulong column = ulong(lane); column < SEISMIC_DIM_W; column += ulong(simd_width)) {
                float q = stage_query(query, row, query_head, column, seismic_words);
                float k = historical
                    ? activation_load(history_key, at3(ulong(token), kv_head, column,
                        SEISMIC_HISTORY_KEY_STRIDE_0, SEISMIC_HISTORY_KEY_STRIDE_1,
                        SEISMIC_HISTORY_KEY_STRIDE_2))
                    : stage_key(prepared_key, ulong(token), kv_head, column, seismic_words);
                partial = metal::fma(q, k, partial);
            }
            float score = simd_sum(partial) * scale;
            float next_maximum = metal::max(maximum, score);
            float carry = metal::exp(maximum - next_maximum);
            float probability = metal::exp(score - next_maximum);
            denominator = denominator * carry + probability;
            for (ulong column = ulong(lane); column < SEISMIC_DIM_W; column += ulong(simd_width)) {
                float v = historical
                    ? activation_load(history_value, at3(ulong(token), kv_head, column,
                        SEISMIC_HISTORY_VALUE_STRIDE_0, SEISMIC_HISTORY_VALUE_STRIDE_1,
                        SEISMIC_HISTORY_VALUE_STRIDE_2))
                    : stage_value(value, ulong(token), kv_head, column, seismic_words);
                ulong at = at3(row, query_head, column, SEISMIC_ACCUMULATOR_STRIDE_0,
                    SEISMIC_ACCUMULATOR_STRIDE_1, SEISMIC_ACCUMULATOR_STRIDE_2);
                accumulator[at] = metal::fma(carry, accumulator[at], probability * v);
            }
            maximum = next_maximum;
        }
    }
    for (ulong column = ulong(lane); column < SEISMIC_DIM_W; column += ulong(simd_width)) {
        ulong at = at3(row, query_head, column, SEISMIC_ACCUMULATOR_STRIDE_0,
            SEISMIC_ACCUMULATOR_STRIDE_1, SEISMIC_ACCUMULATOR_STRIDE_2);
        float attended = denominator > 0.0f ? accumulator[at] / denominator : 0.0f;
        float gate_value = activation_load(gate, at3(row, query_head, column,
            SEISMIC_GATE_STRIDE_0, SEISMIC_GATE_STRIDE_1, SEISMIC_GATE_STRIDE_2));
        activation_store(gated, at3(row, query_head, column, SEISMIC_RESULT_0_STRIDE_0,
            SEISMIC_RESULT_0_STRIDE_1, SEISMIC_RESULT_0_STRIDE_2),
            attended / (1.0f + metal::exp(-gate_value)));
    }
}
