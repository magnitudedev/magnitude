// Convolution and Q/K normalization are performed once per physical row.
// Each row/head owns its complete W-vector, so its normalization has no
// cross-thread dependency. The tail of the grid updates the short window.
inline ulong rpr2(ulong a, ulong b, ulong sa, ulong sb) { return a * sa + b * sb; }
inline ulong rpr3(ulong a, ulong b, ulong c, ulong sa, ulong sb, ulong sc) {
    return a * sa + b * sb + c * sc;
}
inline float rpr_activation(device const uchar *base, ulong logical) {
#if defined(SEISMIC_ELEMENT_A_REPRESENTATION_F32)
    return *reinterpret_cast<device const float *>(base + logical * 4);
#elif defined(SEISMIC_ELEMENT_A_REPRESENTATION_F16)
    return float(*reinterpret_cast<device const half *>(base + logical * 2));
#elif defined(SEISMIC_ELEMENT_A_REPRESENTATION_BF16)
    return as_type<float>(uint(*reinterpret_cast<device const ushort *>(base + logical * 2)) << 16);
#else
#error "recurrent window activation must be dense"
#endif
}
inline void rpr_store(device uchar *base, ulong logical, float value) {
#if defined(SEISMIC_ELEMENT_A_REPRESENTATION_F32)
    *reinterpret_cast<device float *>(base + logical * 4) = value;
#elif defined(SEISMIC_ELEMENT_A_REPRESENTATION_F16)
    *reinterpret_cast<device half *>(base + logical * 2) = half(value);
#elif defined(SEISMIC_ELEMENT_A_REPRESENTATION_BF16)
    uint bits = as_type<uint>(value);
    bits += 0x7fffu + ((bits >> 16) & 1u);
    *reinterpret_cast<device ushort *>(base + logical * 2) = ushort(bits >> 16);
#endif
}
inline float rpr_raw(constant ulong *seismic_words, device const uchar *projection,
    device const float *convolution,
    device const uchar *window, ulong slot, ulong row, ulong local, ulong lo, ulong source_column) {
    float sum = convolution[rpr2(source_column, SEISMIC_DIM_C - 1,
        SEISMIC_CONVOLUTION_STRIDE_0, SEISMIC_CONVOLUTION_STRIDE_1)]
        * rpr_activation(projection, rpr2(row, source_column, SEISMIC_PROJECTION_STRIDE_0,
            SEISMIC_PROJECTION_STRIDE_1));
    for (ulong tap = 0; tap < SEISMIC_DIM_C - 1; ++tap) {
        float previous = local + tap < SEISMIC_DIM_C - 1
            ? rpr_activation(window, rpr3(slot, local + tap, source_column,
                SEISMIC_WINDOW_STRIDE_0, SEISMIC_WINDOW_STRIDE_1, SEISMIC_WINDOW_STRIDE_2))
            : rpr_activation(projection, rpr2(lo + local + tap - (SEISMIC_DIM_C - 1), source_column,
                SEISMIC_PROJECTION_STRIDE_0, SEISMIC_PROJECTION_STRIDE_1));
        sum = metal::fma(convolution[rpr2(source_column, tap,
            SEISMIC_CONVOLUTION_STRIDE_0, SEISMIC_CONVOLUTION_STRIDE_1)], previous, sum);
    }
    return sum / (1.0f + metal::exp(-sum));
}
kernel void qwen_recurrent_prepare(
    device const uchar *projection [[buffer(SEISMIC_BUFFER_PROJECTION)]],
    device const float *convolution [[buffer(SEISMIC_BUFFER_CONVOLUTION)]],
    device const float *rate [[buffer(SEISMIC_BUFFER_RATE)]],
    device const float *time_bias [[buffer(SEISMIC_BUFFER_TIME_BIAS)]],
    device const uchar *recurrent_norm [[buffer(SEISMIC_BUFFER_RECURRENT_NORM)]],
    device const int *segments [[buffer(SEISMIC_BUFFER_SEGMENTS)]],
    device const uchar *window [[buffer(SEISMIC_BUFFER_WINDOW)]],
    device uchar *next_window [[buffer(SEISMIC_RESULT_0_BUFFER)]],
    device uchar *prepared [[buffer(SEISMIC_RESULT_1_BUFFER)]],
    device float *decay_output [[buffer(SEISMIC_RESULT_2_BUFFER)]],
    constant ulong *seismic_words [[buffer(SEISMIC_BUFFER_WORDS)]],
    uint raw_index [[thread_position_in_grid]]) {
    ulong q_heads = 2 * SEISMIC_DIM_NK + SEISMIC_DIM_NV;
    ulong q_width = q_heads * SEISMIC_DIM_W;
    ulong row_threads = SEISMIC_DIM_M * (q_heads + SEISMIC_DIM_NV);
    ulong index = ulong(raw_index);
    if (index < row_threads) {
        ulong row = index / (q_heads + SEISMIC_DIM_NV);
        ulong head = index % (q_heads + SEISMIC_DIM_NV);
        ulong slot = SEISMIC_DIM_B;
        ulong lo = 0;
        for (ulong candidate = 0; candidate < SEISMIC_DIM_B; ++candidate) {
            int low = segments[rpr2(candidate, 0, SEISMIC_SEGMENTS_STRIDE_0,
                SEISMIC_SEGMENTS_STRIDE_1)];
            int high = segments[rpr2(candidate, 1, SEISMIC_SEGMENTS_STRIDE_0,
                SEISMIC_SEGMENTS_STRIDE_1)];
            if (int(row) >= low && int(row) < high) {
                slot = candidate;
                lo = ulong(low);
                break;
            }
        }
        if (head < q_heads) {
            ulong source_first = head * SEISMIC_DIM_W;
            if (slot == SEISMIC_DIM_B) {
                for (ulong column = 0; column < SEISMIC_DIM_W; ++column)
                    rpr_store(prepared, rpr2(row, source_first + column,
                        SEISMIC_RESULT_1_STRIDE_0, SEISMIC_RESULT_1_STRIDE_1), 0.0f);
                return;
            }
            ulong local = row - lo;
            float squares = 0.0f;
            if (head < 2 * SEISMIC_DIM_NK) {
                for (ulong column = 0; column < SEISMIC_DIM_W; ++column) {
                    float value = rpr_raw(seismic_words, projection, convolution, window, slot, row, local,
                        lo, source_first + column);
                    squares = metal::fma(value, value, squares);
                }
            }
            float inverse = head < 2 * SEISMIC_DIM_NK
                ? metal::rsqrt(squares + as_type<float>(uint(SEISMIC_PARAM_PREPARATION_EPSILON)))
                : 1.0f;
            if (head < SEISMIC_DIM_NK) inverse *= metal::rsqrt(float(SEISMIC_DIM_W));
            for (ulong column = 0; column < SEISMIC_DIM_W; ++column) {
                float value = rpr_raw(seismic_words, projection, convolution, window, slot, row, local,
                    lo, source_first + column) * inverse;
                rpr_store(prepared, rpr2(row, source_first + column,
                    SEISMIC_RESULT_1_STRIDE_0, SEISMIC_RESULT_1_STRIDE_1), value);
            }
            return;
        }
        ulong value_head = head - q_heads;
        ulong beta_column = q_width + value_head;
        float beta = 0.0f;
        float decay = 0.0f;
        if (slot != SEISMIC_DIM_B) {
            ulong alpha_column = q_width + SEISMIC_DIM_NV * SEISMIC_DIM_W + value_head;
            ulong beta_input_column = alpha_column + SEISMIC_DIM_NV;
            float alpha = rpr_activation(projection, rpr2(row, alpha_column,
                SEISMIC_PROJECTION_STRIDE_0, SEISMIC_PROJECTION_STRIDE_1));
            float beta_input = rpr_activation(projection, rpr2(row, beta_input_column,
                SEISMIC_PROJECTION_STRIDE_0, SEISMIC_PROJECTION_STRIDE_1));
            beta = 1.0f / (1.0f + metal::exp(-beta_input));
            float shifted = alpha + time_bias[value_head * SEISMIC_TIME_BIAS_STRIDE_0];
            float softplus = metal::max(shifted, 0.0f)
                + metal::log(1.0f + metal::exp(-metal::abs(shifted)));
            decay = metal::exp(rate[value_head * SEISMIC_RATE_STRIDE_0] * softplus);
        }
        rpr_store(prepared, rpr2(row, beta_column, SEISMIC_RESULT_1_STRIDE_0,
            SEISMIC_RESULT_1_STRIDE_1), beta);
        decay_output[rpr2(row, value_head, SEISMIC_RESULT_2_STRIDE_0,
            SEISMIC_RESULT_2_STRIDE_1)] = decay;
        return;
    }
    index -= row_threads;
    ulong window_count = SEISMIC_DIM_B * (SEISMIC_DIM_C - 1) * q_width;
    if (index >= window_count) return;
    ulong slot = index / ((SEISMIC_DIM_C - 1) * q_width);
    ulong rem = index % ((SEISMIC_DIM_C - 1) * q_width);
    ulong tap = rem / q_width;
    ulong column = rem % q_width;
    int low = segments[rpr2(slot, 0, SEISMIC_SEGMENTS_STRIDE_0, SEISMIC_SEGMENTS_STRIDE_1)];
    int high = segments[rpr2(slot, 1, SEISMIC_SEGMENTS_STRIDE_0, SEISMIC_SEGMENTS_STRIDE_1)];
    ulong length = ulong(high - low);
    float value = length + tap < SEISMIC_DIM_C - 1
        ? rpr_activation(window, rpr3(slot, length + tap, column,
            SEISMIC_WINDOW_STRIDE_0, SEISMIC_WINDOW_STRIDE_1, SEISMIC_WINDOW_STRIDE_2))
        : rpr_activation(projection, rpr2(ulong(high) + tap - (SEISMIC_DIM_C - 1), column,
            SEISMIC_PROJECTION_STRIDE_0, SEISMIC_PROJECTION_STRIDE_1));
    rpr_store(next_window, rpr3(slot, tap, column, SEISMIC_RESULT_0_STRIDE_0,
        SEISMIC_RESULT_0_STRIDE_1, SEISMIC_RESULT_0_STRIDE_2), value);
}
