// Only time is serial. A thread owns one value-head/state-row vector across
// all slots, publishes each mixed row once, and writes its final state once.
// Native program admission checks W <= 256 before this kernel is dispatched.
#define QWEN_RECURRENT_WIDTH_CAPACITY 256
inline ulong rs2(ulong a, ulong b, ulong sa, ulong sb) { return a * sa + b * sb; }
inline ulong rs3(ulong a, ulong b, ulong c, ulong sa, ulong sb, ulong sc) {
    return a * sa + b * sb + c * sc;
}
inline ulong rs4(ulong a, ulong b, ulong c, ulong d, ulong sa, ulong sb, ulong sc, ulong sd) {
    return a * sa + b * sb + c * sc + d * sd;
}
inline float rs_activation(device const uchar *base, ulong logical) {
#if defined(SEISMIC_ELEMENT_A_REPRESENTATION_F32)
    return reinterpret_cast<device const float *>(base)[logical];
#elif defined(SEISMIC_ELEMENT_A_REPRESENTATION_F16)
    return float(reinterpret_cast<device const half *>(base)[logical]);
#elif defined(SEISMIC_ELEMENT_A_REPRESENTATION_BF16)
    return as_type<float>(uint(reinterpret_cast<device const ushort *>(base)[logical]) << 16);
#endif
}
inline void rs_store_activation(device uchar *base, ulong logical, float value) {
#if defined(SEISMIC_ELEMENT_A_REPRESENTATION_F32)
    reinterpret_cast<device float *>(base)[logical] = value;
#elif defined(SEISMIC_ELEMENT_A_REPRESENTATION_F16)
    reinterpret_cast<device half *>(base)[logical] = half(value);
#elif defined(SEISMIC_ELEMENT_A_REPRESENTATION_BF16)
    uint bits = as_type<uint>(value);
    bits += 0x7fffu + ((bits >> 16) & 1u);
    reinterpret_cast<device ushort *>(base)[logical] = ushort(bits >> 16);
#endif
}
kernel void qwen_recurrent_scan(
    device const uchar *prepared [[buffer(SEISMIC_BUFFER_PREPARED)]],
    device const float *decay [[buffer(SEISMIC_BUFFER_DECAY)]],
    device const int *segments [[buffer(SEISMIC_BUFFER_SEGMENTS)]],
    device const float *delta [[buffer(SEISMIC_BUFFER_DELTA)]],
    device float *next_delta [[buffer(SEISMIC_RESULT_0_BUFFER)]],
    device uchar *mixed [[buffer(SEISMIC_RESULT_1_BUFFER)]],
    constant ulong *seismic_words [[buffer(SEISMIC_BUFFER_WORDS)]],
    uint raw_index [[thread_position_in_grid]]) {
    ulong index = ulong(raw_index);
    if (index >= SEISMIC_DIM_NV * SEISMIC_DIM_W) return;
    ulong value_head = index / SEISMIC_DIM_W;
    ulong state_row = index % SEISMIC_DIM_W;
    ulong q_width = (2 * SEISMIC_DIM_NK + SEISMIC_DIM_NV) * SEISMIC_DIM_W;
    bool grouped = SEISMIC_PARAM_GROUPED != 0;
    ulong key_head = grouped ? value_head * SEISMIC_DIM_NK / SEISMIC_DIM_NV
        : value_head % SEISMIC_DIM_NK;
    for (ulong row = 0; row < SEISMIC_DIM_M; ++row)
        rs_store_activation(mixed, rs3(row, value_head, state_row, SEISMIC_RESULT_1_STRIDE_0,
            SEISMIC_RESULT_1_STRIDE_1, SEISMIC_RESULT_1_STRIDE_2), 0.0f);
    for (ulong slot = 0; slot < SEISMIC_DIM_B; ++slot) {
        thread float state[QWEN_RECURRENT_WIDTH_CAPACITY];
        for (ulong column = 0; column < SEISMIC_DIM_W; ++column)
            state[column] = delta[rs4(slot, value_head, state_row, column,
                SEISMIC_DELTA_STRIDE_0, SEISMIC_DELTA_STRIDE_1,
                SEISMIC_DELTA_STRIDE_2, SEISMIC_DELTA_STRIDE_3)];
        int lo = segments[rs2(slot, 0, SEISMIC_SEGMENTS_STRIDE_0, SEISMIC_SEGMENTS_STRIDE_1)];
        int hi = segments[rs2(slot, 1, SEISMIC_SEGMENTS_STRIDE_0, SEISMIC_SEGMENTS_STRIDE_1)];
        for (int token = lo; token < hi; ++token) {
            ulong row = ulong(token);
            float factor = decay[rs2(row, value_head,
                SEISMIC_DECAY_STRIDE_0, SEISMIC_DECAY_STRIDE_1)];
            float beta = rs_activation(prepared, rs2(row, q_width + value_head,
                SEISMIC_PREPARED_STRIDE_0, SEISMIC_PREPARED_STRIDE_1));
            float remembered = 0.0f;
            for (ulong column = 0; column < SEISMIC_DIM_W; ++column) {
                float key = rs_activation(prepared, rs2(row, (SEISMIC_DIM_NK + key_head) * SEISMIC_DIM_W + column,
                    SEISMIC_PREPARED_STRIDE_0, SEISMIC_PREPARED_STRIDE_1));
                remembered = metal::fma(state[column] * factor, key, remembered);
            }
            float value = rs_activation(prepared, rs2(row, (2 * SEISMIC_DIM_NK + value_head) * SEISMIC_DIM_W
                + state_row, SEISMIC_PREPARED_STRIDE_0, SEISMIC_PREPARED_STRIDE_1));
            float residual = (value - remembered) * beta;
            float output = 0.0f;
            for (ulong column = 0; column < SEISMIC_DIM_W; ++column) {
                float key = rs_activation(prepared, rs2(row, (SEISMIC_DIM_NK + key_head) * SEISMIC_DIM_W + column,
                    SEISMIC_PREPARED_STRIDE_0, SEISMIC_PREPARED_STRIDE_1));
                state[column] = metal::fma(residual, key, state[column] * factor);
                float query = rs_activation(prepared, rs2(row, key_head * SEISMIC_DIM_W + column,
                    SEISMIC_PREPARED_STRIDE_0, SEISMIC_PREPARED_STRIDE_1));
                output = metal::fma(state[column], query, output);
            }
            rs_store_activation(mixed, rs3(row, value_head, state_row, SEISMIC_RESULT_1_STRIDE_0,
                SEISMIC_RESULT_1_STRIDE_1, SEISMIC_RESULT_1_STRIDE_2), output);
        }
        for (ulong column = 0; column < SEISMIC_DIM_W; ++column)
            next_delta[rs4(slot, value_head, state_row, column,
                SEISMIC_RESULT_0_STRIDE_0, SEISMIC_RESULT_0_STRIDE_1,
                SEISMIC_RESULT_0_STRIDE_2, SEISMIC_RESULT_0_STRIDE_3)] = state[column];
    }
}
