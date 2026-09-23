inline ulong at2(ulong a, ulong b, ulong sa, ulong sb) { return a * sa + b * sb; }
inline ulong at3(ulong a, ulong b, ulong c, ulong sa, ulong sb, ulong sc) {
    return a * sa + b * sb + c * sc;
}
inline uint packed_code(device const uchar *bytes, ulong bit, uint width) {
    uint value = 0;
    for (uint offset = 0; offset < width; ++offset) value |= uint((bytes[(bit + offset) >> 3] >> ((bit + offset) & 7)) & 1) << offset;
    return value;
}
inline float packet(device const uchar *base, ulong logical, uint kind, ulong size, ulong group,
    ulong words, ulong coefficients, ulong factor, ulong bias) {
    if (kind == 0) return *reinterpret_cast<device const float *>(base + logical * size);
    if (kind == 1) return float(*reinterpret_cast<device const half *>(base + logical * size));
    if (kind == 2) return as_type<float>(uint(*reinterpret_cast<device const ushort *>(base + logical * size)) << 16);
    device const uchar *resident = base + (logical / group) * size;
    ulong position = logical % group;
    if (kind == 8) return float(int(reinterpret_cast<device const char *>(resident + words)[position])) * float(*reinterpret_cast<device const half *>(resident + factor));
    if (kind == 14) {
        const int table[16] = {-127,-104,-83,-65,-49,-35,-22,-10,1,13,25,38,53,69,89,113};
        return reinterpret_cast<device const float *>(resident + factor)[position / 32] * float(table[packed_code(resident + words, position * 4, 4)]);
    }
    int code = int(packed_code(resident + words, position * kind, kind));
    if (kind == 6) code -= 32;
    ulong ci = position / (kind == 6 ? 16 : 32);
    if (kind == 6) return float(code * int(reinterpret_cast<device const char *>(resident + coefficients)[ci])) * float(*reinterpret_cast<device const half *>(resident + factor));
    uint scale_code = packed_code(resident + coefficients, ci * 12, 6);
    uint bias_code = packed_code(resident + coefficients, ci * 12 + 6, 6);
    return metal::fma(float(*reinterpret_cast<device const half *>(resident + factor)) * float(scale_code), float(code), -float(*reinterpret_cast<device const half *>(resident + bias)) * float(bias_code));
}
#define DENSE_CASE(PREFIX, KIND) packet(base, logical, KIND, PREFIX##_PACKET_SIZE, 1, 0, 0, 0, 0)
#define Q8_CASE(PREFIX) packet(base, logical, 8, PREFIX##_PACKET_SIZE, PREFIX##_LOGICAL_GROUP, PREFIX##_PLANE_0_OFFSET, 0, PREFIX##_PLANE_1_OFFSET, 0)
#define QK_CASE(PREFIX, KIND) packet(base, logical, KIND, PREFIX##_PACKET_SIZE, PREFIX##_LOGICAL_GROUP, PREFIX##_PLANE_0_OFFSET, PREFIX##_PLANE_1_OFFSET, PREFIX##_PLANE_2_OFFSET, PREFIX##_PLANE_3_OFFSET)
#define Q6_CASE(PREFIX) packet(base, logical, 6, PREFIX##_PACKET_SIZE, PREFIX##_LOGICAL_GROUP, PREFIX##_PLANE_0_OFFSET, PREFIX##_PLANE_1_OFFSET, PREFIX##_PLANE_2_OFFSET, 0)
#define IQ_CASE(PREFIX) packet(base, logical, 14, PREFIX##_PACKET_SIZE, PREFIX##_LOGICAL_GROUP, PREFIX##_PLANE_0_OFFSET, 0, PREFIX##_PLANE_1_OFFSET, 0)
inline float load_input_norm(device const uchar *base, ulong logical) {
#if defined(SEISMIC_INPUT_NORM_REPRESENTATION_F32)
    return DENSE_CASE(SEISMIC_INPUT_NORM, 0);
#elif defined(SEISMIC_INPUT_NORM_REPRESENTATION_F16)
    return DENSE_CASE(SEISMIC_INPUT_NORM, 1);
#elif defined(SEISMIC_INPUT_NORM_REPRESENTATION_BF16)
    return DENSE_CASE(SEISMIC_INPUT_NORM, 2);
#else
#error "qwen_attention_rows requires a dense input norm"
#endif
}
#define DEFINE_WEIGHT_LOAD(NAME, PREFIX) \
inline float NAME(device const uchar *base, ulong logical) { \
    return packet(base, logical, PREFIX##_KIND, PREFIX##_PACKET_SIZE, PREFIX##_GROUP, PREFIX##_P0, PREFIX##_P1, PREFIX##_P2, PREFIX##_P3); \
}
#if defined(SEISMIC_QUERY_GATE_WEIGHT_REPRESENTATION_F32)
#define QGW_KIND 0
#define QGW_GROUP 1
#define QGW_P0 0
#define QGW_P1 0
#define QGW_P2 0
#define QGW_P3 0
#elif defined(SEISMIC_QUERY_GATE_WEIGHT_REPRESENTATION_F16)
#define QGW_KIND 1
#define QGW_GROUP 1
#define QGW_P0 0
#define QGW_P1 0
#define QGW_P2 0
#define QGW_P3 0
#elif defined(SEISMIC_QUERY_GATE_WEIGHT_REPRESENTATION_BF16)
#define QGW_KIND 2
#define QGW_GROUP 1
#define QGW_P0 0
#define QGW_P1 0
#define QGW_P2 0
#define QGW_P3 0
#elif defined(SEISMIC_QUERY_GATE_WEIGHT_REPRESENTATION_Q8G32S)
#define QGW_KIND 8
#define QGW_GROUP SEISMIC_QUERY_GATE_WEIGHT_LOGICAL_GROUP
#define QGW_P0 SEISMIC_QUERY_GATE_WEIGHT_PLANE_0_OFFSET
#define QGW_P1 0
#define QGW_P2 SEISMIC_QUERY_GATE_WEIGHT_PLANE_1_OFFSET
#define QGW_P3 0
#elif defined(SEISMIC_QUERY_GATE_WEIGHT_REPRESENTATION_Q4K)
#define QGW_KIND 4
#define QGW_GROUP SEISMIC_QUERY_GATE_WEIGHT_LOGICAL_GROUP
#define QGW_P0 SEISMIC_QUERY_GATE_WEIGHT_PLANE_0_OFFSET
#define QGW_P1 SEISMIC_QUERY_GATE_WEIGHT_PLANE_1_OFFSET
#define QGW_P2 SEISMIC_QUERY_GATE_WEIGHT_PLANE_2_OFFSET
#define QGW_P3 SEISMIC_QUERY_GATE_WEIGHT_PLANE_3_OFFSET
#elif defined(SEISMIC_QUERY_GATE_WEIGHT_REPRESENTATION_Q5K)
#define QGW_KIND 5
#define QGW_GROUP SEISMIC_QUERY_GATE_WEIGHT_LOGICAL_GROUP
#define QGW_P0 SEISMIC_QUERY_GATE_WEIGHT_PLANE_0_OFFSET
#define QGW_P1 SEISMIC_QUERY_GATE_WEIGHT_PLANE_1_OFFSET
#define QGW_P2 SEISMIC_QUERY_GATE_WEIGHT_PLANE_2_OFFSET
#define QGW_P3 SEISMIC_QUERY_GATE_WEIGHT_PLANE_3_OFFSET
#elif defined(SEISMIC_QUERY_GATE_WEIGHT_REPRESENTATION_Q6K)
#define QGW_KIND 6
#define QGW_GROUP SEISMIC_QUERY_GATE_WEIGHT_LOGICAL_GROUP
#define QGW_P0 SEISMIC_QUERY_GATE_WEIGHT_PLANE_0_OFFSET
#define QGW_P1 SEISMIC_QUERY_GATE_WEIGHT_PLANE_1_OFFSET
#define QGW_P2 SEISMIC_QUERY_GATE_WEIGHT_PLANE_2_OFFSET
#define QGW_P3 0
#elif defined(SEISMIC_QUERY_GATE_WEIGHT_REPRESENTATION_IQ4G32)
#define QGW_KIND 14
#define QGW_GROUP SEISMIC_QUERY_GATE_WEIGHT_LOGICAL_GROUP
#define QGW_P0 SEISMIC_QUERY_GATE_WEIGHT_PLANE_0_OFFSET
#define QGW_P1 0
#define QGW_P2 SEISMIC_QUERY_GATE_WEIGHT_PLANE_1_OFFSET
#define QGW_P3 0
#else
#error "unsupported attention query weight"
#endif
#define QGW_PACKET_SIZE SEISMIC_QUERY_GATE_WEIGHT_PACKET_SIZE
DEFINE_WEIGHT_LOAD(load_query_weight, QGW)
inline float load_key_weight(device const uchar *base, ulong logical) {
#if defined(SEISMIC_KEY_WEIGHT_REPRESENTATION_F32)
    return DENSE_CASE(SEISMIC_KEY_WEIGHT, 0);
#elif defined(SEISMIC_KEY_WEIGHT_REPRESENTATION_F16)
    return DENSE_CASE(SEISMIC_KEY_WEIGHT, 1);
#elif defined(SEISMIC_KEY_WEIGHT_REPRESENTATION_BF16)
    return DENSE_CASE(SEISMIC_KEY_WEIGHT, 2);
#elif defined(SEISMIC_KEY_WEIGHT_REPRESENTATION_Q8G32S)
    return Q8_CASE(SEISMIC_KEY_WEIGHT);
#elif defined(SEISMIC_KEY_WEIGHT_REPRESENTATION_Q4K)
    return QK_CASE(SEISMIC_KEY_WEIGHT, 4);
#elif defined(SEISMIC_KEY_WEIGHT_REPRESENTATION_Q5K)
    return QK_CASE(SEISMIC_KEY_WEIGHT, 5);
#elif defined(SEISMIC_KEY_WEIGHT_REPRESENTATION_Q6K)
    return Q6_CASE(SEISMIC_KEY_WEIGHT);
#elif defined(SEISMIC_KEY_WEIGHT_REPRESENTATION_IQ4G32)
    return IQ_CASE(SEISMIC_KEY_WEIGHT);
#else
#error "unsupported attention key weight"
#endif
}
inline float load_value_weight(device const uchar *base, ulong logical) {
#if defined(SEISMIC_VALUE_WEIGHT_REPRESENTATION_F32)
    return DENSE_CASE(SEISMIC_VALUE_WEIGHT, 0);
#elif defined(SEISMIC_VALUE_WEIGHT_REPRESENTATION_F16)
    return DENSE_CASE(SEISMIC_VALUE_WEIGHT, 1);
#elif defined(SEISMIC_VALUE_WEIGHT_REPRESENTATION_BF16)
    return DENSE_CASE(SEISMIC_VALUE_WEIGHT, 2);
#elif defined(SEISMIC_VALUE_WEIGHT_REPRESENTATION_Q8G32S)
    return Q8_CASE(SEISMIC_VALUE_WEIGHT);
#elif defined(SEISMIC_VALUE_WEIGHT_REPRESENTATION_Q4K)
    return QK_CASE(SEISMIC_VALUE_WEIGHT, 4);
#elif defined(SEISMIC_VALUE_WEIGHT_REPRESENTATION_Q5K)
    return QK_CASE(SEISMIC_VALUE_WEIGHT, 5);
#elif defined(SEISMIC_VALUE_WEIGHT_REPRESENTATION_Q6K)
    return Q6_CASE(SEISMIC_VALUE_WEIGHT);
#elif defined(SEISMIC_VALUE_WEIGHT_REPRESENTATION_IQ4G32)
    return IQ_CASE(SEISMIC_VALUE_WEIGHT);
#else
#error "unsupported attention value weight"
#endif
}
inline float load_output_weight(device const uchar *base, ulong logical) {
#if defined(SEISMIC_OUTPUT_WEIGHT_REPRESENTATION_F32)
    return DENSE_CASE(SEISMIC_OUTPUT_WEIGHT, 0);
#elif defined(SEISMIC_OUTPUT_WEIGHT_REPRESENTATION_F16)
    return DENSE_CASE(SEISMIC_OUTPUT_WEIGHT, 1);
#elif defined(SEISMIC_OUTPUT_WEIGHT_REPRESENTATION_BF16)
    return DENSE_CASE(SEISMIC_OUTPUT_WEIGHT, 2);
#elif defined(SEISMIC_OUTPUT_WEIGHT_REPRESENTATION_Q8G32S)
    return Q8_CASE(SEISMIC_OUTPUT_WEIGHT);
#elif defined(SEISMIC_OUTPUT_WEIGHT_REPRESENTATION_Q4K)
    return QK_CASE(SEISMIC_OUTPUT_WEIGHT, 4);
#elif defined(SEISMIC_OUTPUT_WEIGHT_REPRESENTATION_Q5K)
    return QK_CASE(SEISMIC_OUTPUT_WEIGHT, 5);
#elif defined(SEISMIC_OUTPUT_WEIGHT_REPRESENTATION_Q6K)
    return Q6_CASE(SEISMIC_OUTPUT_WEIGHT);
#elif defined(SEISMIC_OUTPUT_WEIGHT_REPRESENTATION_IQ4G32)
    return IQ_CASE(SEISMIC_OUTPUT_WEIGHT);
#else
#error "unsupported attention output weight"
#endif
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
inline float hidden_inverse(device const float *hidden, constant ulong *seismic_words,
    ulong row, float epsilon) {
    float squares = 0.0f;
    for (ulong source = 0; source < SEISMIC_DIM_D; ++source) {
        float value = hidden[at2(row, source, SEISMIC_HIDDEN_STRIDE_0, SEISMIC_HIDDEN_STRIDE_1)];
        squares += value * value;
    }
    return metal::rsqrt(squares / float(SEISMIC_DIM_D) + epsilon);
}
inline float normalized_hidden(device const float *hidden, device const uchar *input_norm,
    constant ulong *seismic_words, ulong row, ulong source, float inverse) {
    return hidden[at2(row, source, SEISMIC_HIDDEN_STRIDE_0, SEISMIC_HIDDEN_STRIDE_1)] * inverse
        * load_input_norm(input_norm, source * SEISMIC_INPUT_NORM_STRIDE_0);
}
inline float projection(device const float *hidden, device const uchar *input_norm,
    device const uchar *weight, uint weight_kind,
    constant ulong *seismic_words, ulong row, ulong output, float inverse) {
    float sum = 0.0f;
    for (ulong source = 0; source < SEISMIC_DIM_D; ++source) {
        ulong logical = output * SEISMIC_DIM_D + source;
        float weight_value = weight_kind == 0 ? load_query_weight(weight, logical)
            : (weight_kind == 1 ? load_key_weight(weight, logical) : load_value_weight(weight, logical));
        sum = metal::fma(normalized_hidden(hidden, input_norm, seismic_words, row, source, inverse),
            weight_value, sum);
    }
    return sum;
}
inline float norm_rotary(device const float *hidden, device const uchar *input_norm,
    device const uchar *weight, uint weight_kind,
    device const float *norm, device const int *coordinates, device const int *components,
    constant ulong *seismic_words, ulong row, ulong head, ulong column, float inverse,
    float base, float epsilon) {
    ulong width = 2 * SEISMIC_DIM_P + SEISMIC_DIM_S;
    ulong first = head * width;
    float squares = 0.0f;
    for (ulong i = 0; i < width; ++i) {
        float value = projection(hidden, input_norm, weight, weight_kind,
            seismic_words, row, first + i, inverse);
        squares += value * value;
    }
    float value = projection(hidden, input_norm, weight, weight_kind,
        seismic_words, row, first + column, inverse);
    float normalized = value * metal::rsqrt(squares / float(width) + epsilon)
        * norm[column * (norm == nullptr ? 0 : 1)];
    if (column >= 2 * SEISMIC_DIM_P) return normalized;
    ulong pair = column % SEISMIC_DIM_P;
    ulong paired = column < SEISMIC_DIM_P ? column + SEISMIC_DIM_P : column - SEISMIC_DIM_P;
    float other = projection(hidden, input_norm, weight, weight_kind,
        seismic_words, row, first + paired, inverse) * metal::rsqrt(squares / float(width) + epsilon)
        * norm[paired];
    int component = components[pair * SEISMIC_ROTARY_COMPONENTS_STRIDE_0];
    int coordinate = coordinates[at2(row, ulong(component), SEISMIC_COORDINATES_STRIDE_0,
        SEISMIC_COORDINATES_STRIDE_1)];
    float angle = float(coordinate) * metal::exp(-metal::log(base) * float(2 * pair)
        / float(2 * SEISMIC_DIM_P));
    return column < SEISMIC_DIM_P
        ? normalized * metal::cos(angle) - other * metal::sin(angle)
        : normalized * metal::cos(angle) + other * metal::sin(angle);
}
inline float query_value(device const float *hidden, device const uchar *input_norm,
    device const uchar *query_gate_weight, device const float *query_norm,
    device const int *coordinates, device const int *components, ulong row, ulong query_head,
    ulong column, float inverse, float base, float epsilon, constant ulong *seismic_words) {
    ulong width = 2 * SEISMIC_DIM_P + SEISMIC_DIM_S;
    return norm_rotary(hidden, input_norm, query_gate_weight,
        0, query_norm, coordinates, components,
        seismic_words, row, query_head * 2, column, inverse, base, epsilon);
}
inline float fresh_key_value(device const float *hidden, device const uchar *input_norm,
    device const uchar *key_weight, device const float *key_norm,
    device const int *coordinates, device const int *components, ulong row, ulong kv_head,
    ulong column, float inverse, float base, float epsilon, constant ulong *seismic_words) {
    return norm_rotary(hidden, input_norm, key_weight, 1, key_norm, coordinates,
        components, seismic_words, row, kv_head, column, inverse, base, epsilon);
}
inline float fresh_value_at(device const float *hidden, device const uchar *input_norm,
    device const uchar *value_weight, constant ulong *seismic_words, ulong row, ulong kv_head,
    ulong column, float inverse) {
    ulong width = 2 * SEISMIC_DIM_P + SEISMIC_DIM_S;
    return projection(hidden, input_norm, value_weight, 2,
        seismic_words, row, kv_head * width + column, inverse);
}
kernel void qwen_attention_rows(
    device const float *hidden [[buffer(SEISMIC_BUFFER_HIDDEN)]],
    device const uchar *input_norm [[buffer(SEISMIC_BUFFER_INPUT_NORM)]],
    device const uchar *query_gate_weight [[buffer(SEISMIC_BUFFER_QUERY_GATE_WEIGHT)]],
    device const uchar *key_weight [[buffer(SEISMIC_BUFFER_KEY_WEIGHT)]],
    device const uchar *value_weight [[buffer(SEISMIC_BUFFER_VALUE_WEIGHT)]],
    device const float *query_norm [[buffer(SEISMIC_BUFFER_QUERY_NORM)]],
    device const float *key_norm [[buffer(SEISMIC_BUFFER_KEY_NORM)]],
    device const uchar *output_weight [[buffer(SEISMIC_BUFFER_OUTPUT_WEIGHT)]],
    device const int *coordinates [[buffer(SEISMIC_BUFFER_COORDINATES)]],
    device const int *rotary_components [[buffer(SEISMIC_BUFFER_ROTARY_COMPONENTS)]],
    device const int *visible [[buffer(SEISMIC_BUFFER_VISIBLE)]],
    device const int *fresh [[buffer(SEISMIC_BUFFER_FRESH)]],
    device const int *destinations [[buffer(SEISMIC_BUFFER_DESTINATIONS)]],
    device uchar *history_key [[buffer(SEISMIC_BUFFER_HISTORY_KEY)]],
    device uchar *history_value [[buffer(SEISMIC_BUFFER_HISTORY_VALUE)]],
    device float *result [[buffer(SEISMIC_RESULT_0_BUFFER)]],
    constant ulong *seismic_words [[buffer(SEISMIC_BUFFER_WORDS)]],
    uint raw_index [[thread_position_in_grid]]) {
    ulong index = ulong(raw_index);
    ulong output_count = SEISMIC_DIM_M * SEISMIC_DIM_D;
    ulong width = 2 * SEISMIC_DIM_P + SEISMIC_DIM_S;
    float base = as_type<float>(uint(SEISMIC_PARAM_BASE));
    float epsilon = as_type<float>(uint(SEISMIC_PARAM_EPSILON));
    float scale = as_type<float>(uint(SEISMIC_PARAM_SCALE));
    if (index < output_count) {
        ulong row = index / SEISMIC_DIM_D;
        ulong output = index % SEISMIC_DIM_D;
        float inverse = hidden_inverse(hidden, seismic_words, row, epsilon);
        float projected = 0.0f;
        for (ulong query_head = 0; query_head < SEISMIC_DIM_KV * SEISMIC_DIM_G; ++query_head) {
            ulong kv_head = query_head / SEISMIC_DIM_G;
            for (ulong value_column = 0; value_column < width; ++value_column) {
                float maximum = -INFINITY;
                float denominator = 0.0f;
                float accumulator = 0.0f;
                for (ulong span = 0; span < SEISMIC_DIM_R + 1; ++span) {
                    int lo = span < SEISMIC_DIM_R
                        ? visible[at3(row, span, 0, SEISMIC_VISIBLE_STRIDE_0,
                            SEISMIC_VISIBLE_STRIDE_1, SEISMIC_VISIBLE_STRIDE_2)]
                        : fresh[at2(row, 0, SEISMIC_FRESH_STRIDE_0, SEISMIC_FRESH_STRIDE_1)];
                    int hi = span < SEISMIC_DIM_R
                        ? visible[at3(row, span, 1, SEISMIC_VISIBLE_STRIDE_0,
                            SEISMIC_VISIBLE_STRIDE_1, SEISMIC_VISIBLE_STRIDE_2)]
                        : fresh[at2(row, 1, SEISMIC_FRESH_STRIDE_0, SEISMIC_FRESH_STRIDE_1)];
                    if (hi <= lo) continue;
                    float span_maximum = -INFINITY;
                    for (int token = lo; token < hi; ++token) {
                        float score = 0.0f;
                        float token_inverse = span < SEISMIC_DIM_R ? 0.0f
                            : hidden_inverse(hidden, seismic_words, ulong(token), epsilon);
                        for (ulong column = 0; column < width; ++column) {
                            float q = query_value(hidden, input_norm, query_gate_weight, query_norm,
                                coordinates, rotary_components, row, query_head, column, inverse,
                                base, epsilon, seismic_words);
                            float k = span < SEISMIC_DIM_R
                                ? activation_load(history_key, at3(ulong(token), kv_head, column,
                                    SEISMIC_HISTORY_KEY_STRIDE_0, SEISMIC_HISTORY_KEY_STRIDE_1,
                                    SEISMIC_HISTORY_KEY_STRIDE_2))
                                : fresh_key_value(hidden, input_norm, key_weight, key_norm,
                                    coordinates, rotary_components, ulong(token), kv_head, column,
                                    token_inverse, base, epsilon, seismic_words);
                            score += q * k;
                        }
                        span_maximum = metal::max(span_maximum, score * scale);
                    }
                    float next_maximum = metal::max(maximum, span_maximum);
                    float carry = metal::exp(maximum - next_maximum);
                    denominator *= carry;
                    accumulator *= carry;
                    for (int token = lo; token < hi; ++token) {
                        float score = 0.0f;
                        float token_inverse = span < SEISMIC_DIM_R ? 0.0f
                            : hidden_inverse(hidden, seismic_words, ulong(token), epsilon);
                        for (ulong column = 0; column < width; ++column) {
                            float q = query_value(hidden, input_norm, query_gate_weight, query_norm,
                                coordinates, rotary_components, row, query_head, column, inverse,
                                base, epsilon, seismic_words);
                            float k = span < SEISMIC_DIM_R
                                ? activation_load(history_key, at3(ulong(token), kv_head, column,
                                    SEISMIC_HISTORY_KEY_STRIDE_0, SEISMIC_HISTORY_KEY_STRIDE_1,
                                    SEISMIC_HISTORY_KEY_STRIDE_2))
                                : fresh_key_value(hidden, input_norm, key_weight, key_norm,
                                    coordinates, rotary_components, ulong(token), kv_head, column,
                                    token_inverse, base, epsilon, seismic_words);
                            score += q * k;
                        }
                        float probability = metal::exp(score * scale - next_maximum);
                        float value = span < SEISMIC_DIM_R
                            ? activation_load(history_value, at3(ulong(token), kv_head, value_column,
                                SEISMIC_HISTORY_VALUE_STRIDE_0, SEISMIC_HISTORY_VALUE_STRIDE_1,
                                SEISMIC_HISTORY_VALUE_STRIDE_2))
                            : fresh_value_at(hidden, input_norm, value_weight, seismic_words,
                                ulong(token), kv_head, value_column, token_inverse);
                        denominator += probability;
                        accumulator = metal::fma(probability, value, accumulator);
                    }
                    maximum = next_maximum;
                }
                float attended = denominator > 0.0f ? accumulator / denominator : 0.0f;
                ulong gate_output = query_head * 2 * width + width + value_column;
                float gate = projection(hidden, input_norm, query_gate_weight, 0,
                    seismic_words, row, gate_output, inverse);
                float gated = attended * (1.0f / (1.0f + metal::exp(-gate)));
                ulong projected_input = query_head * width + value_column;
                projected = metal::fma(gated, load_output_weight(output_weight,
                    output * (SEISMIC_DIM_KV * SEISMIC_DIM_G * width) + projected_input), projected);
            }
        }
        result[at2(row, output, SEISMIC_RESULT_0_STRIDE_0, SEISMIC_RESULT_0_STRIDE_1)] =
            hidden[at2(row, output, SEISMIC_HIDDEN_STRIDE_0, SEISMIC_HIDDEN_STRIDE_1)] + projected;
        return;
    }
    index -= output_count;
    ulong state_count = SEISMIC_DIM_M * SEISMIC_DIM_KV * width;
    if (index >= state_count) return;
    ulong row = index / (SEISMIC_DIM_KV * width);
    ulong rem = index % (SEISMIC_DIM_KV * width);
    ulong kv_head = rem / width;
    ulong column = rem % width;
    int destination = destinations[row * SEISMIC_DESTINATIONS_STRIDE_0];
    if (destination < 0) return;
    float inverse = hidden_inverse(hidden, seismic_words, row, epsilon);
    activation_store(history_key, at3(ulong(destination), kv_head, column, SEISMIC_HISTORY_KEY_STRIDE_0,
        SEISMIC_HISTORY_KEY_STRIDE_1, SEISMIC_HISTORY_KEY_STRIDE_2),
        fresh_key_value(hidden, input_norm, key_weight, key_norm, coordinates,
            rotary_components, row, kv_head, column, inverse, base, epsilon, seismic_words));
    activation_store(history_value, at3(ulong(destination), kv_head, column, SEISMIC_HISTORY_VALUE_STRIDE_0,
        SEISMIC_HISTORY_VALUE_STRIDE_1, SEISMIC_HISTORY_VALUE_STRIDE_2),
        fresh_value_at(hidden, input_norm, value_weight, seismic_words, row, kv_head, column, inverse));
}
