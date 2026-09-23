inline ulong at2(ulong a, ulong b, ulong sa, ulong sb) { return a * sa + b * sb; }
inline ulong at3(ulong a, ulong b, ulong c, ulong sa, ulong sb, ulong sc) {
    return a * sa + b * sb + c * sc;
}
inline uint packed_code(device const uchar *bytes, ulong bit, uint width) {
    ulong byte = bit >> 3;
    uint shift = uint(bit & 7);
    uint value = uint(bytes[byte]) >> shift;
    if (shift + width > 8) value |= uint(bytes[byte + 1]) << (8 - shift);
    return value & ((1u << width) - 1u);
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
inline float load_query_weight(device const uchar *base, ulong logical) {
#if defined(SEISMIC_QUERY_GATE_WEIGHT_REPRESENTATION_F32)
    return DENSE_CASE(SEISMIC_QUERY_GATE_WEIGHT, 0);
#elif defined(SEISMIC_QUERY_GATE_WEIGHT_REPRESENTATION_F16)
    return DENSE_CASE(SEISMIC_QUERY_GATE_WEIGHT, 1);
#elif defined(SEISMIC_QUERY_GATE_WEIGHT_REPRESENTATION_BF16)
    return DENSE_CASE(SEISMIC_QUERY_GATE_WEIGHT, 2);
#elif defined(SEISMIC_QUERY_GATE_WEIGHT_REPRESENTATION_Q8G32S)
    return Q8_CASE(SEISMIC_QUERY_GATE_WEIGHT);
#elif defined(SEISMIC_QUERY_GATE_WEIGHT_REPRESENTATION_Q4K)
    return QK_CASE(SEISMIC_QUERY_GATE_WEIGHT, 4);
#elif defined(SEISMIC_QUERY_GATE_WEIGHT_REPRESENTATION_Q5K)
    return QK_CASE(SEISMIC_QUERY_GATE_WEIGHT, 5);
#elif defined(SEISMIC_QUERY_GATE_WEIGHT_REPRESENTATION_Q6K)
    return Q6_CASE(SEISMIC_QUERY_GATE_WEIGHT);
#elif defined(SEISMIC_QUERY_GATE_WEIGHT_REPRESENTATION_IQ4G32)
    return IQ_CASE(SEISMIC_QUERY_GATE_WEIGHT);
#else
#error "unsupported attention query weight"
#endif
}
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
kernel void qwen_attention_project(
    device const uchar *normalized [[buffer(SEISMIC_BUFFER_NORMALIZED)]],
    device const float *query_norm [[buffer(SEISMIC_BUFFER_QUERY_NORM)]],
    device const uchar *query_gate_weight [[buffer(SEISMIC_BUFFER_QUERY_GATE_WEIGHT)]],
    device const uchar *key_weight [[buffer(SEISMIC_BUFFER_KEY_WEIGHT)]],
    device const uchar *value_weight [[buffer(SEISMIC_BUFFER_VALUE_WEIGHT)]],
    device uchar *query_gate [[buffer(SEISMIC_RESULT_0_BUFFER)]],
    device uchar *key [[buffer(SEISMIC_RESULT_1_BUFFER)]],
    device uchar *value [[buffer(SEISMIC_RESULT_2_BUFFER)]],
    constant ulong *seismic_words [[buffer(SEISMIC_BUFFER_WORDS)]],
    uint raw_index [[thread_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd_width [[threads_per_simdgroup]]) {
    // One SIMD group owns one weight vector and up to eight rows. Each lane
    // decodes a weight once and applies it to all live rows in registers.
    const ulong row_tiles = (SEISMIC_DIM_M + 7) / 8;
    const ulong query_width = SEISMIC_DIM_KV * SEISMIC_DIM_G * 2 * SEISMIC_DIM_W;
    const ulong key_width = SEISMIC_DIM_KV * SEISMIC_DIM_W;
    const ulong output_width = query_width + 2 * key_width;
    ulong dot = ulong(raw_index) / ulong(simd_width);
    if (dot >= row_tiles * output_width) return;
    ulong output_index = dot / row_tiles;
    ulong row_base = (dot % row_tiles) * 8;
    uint kind = output_index < query_width ? 0
        : (output_index < query_width + key_width ? 1 : 2);
    ulong output = kind == 0 ? output_index
        : (kind == 1 ? output_index - query_width
            : output_index - query_width - key_width);
    float sum[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    for (ulong source = ulong(lane); source < SEISMIC_DIM_D; source += ulong(simd_width)) {
        ulong logical = output * SEISMIC_DIM_D + source;
        float weight = kind == 0 ? load_query_weight(query_gate_weight, logical)
            : (kind == 1 ? load_key_weight(key_weight, logical)
                : load_value_weight(value_weight, logical));
        for (uint r = 0; r < 8; ++r) {
            ulong row = row_base + r;
            if (row < SEISMIC_DIM_M) {
                float input = activation_load(normalized, at2(row, source,
                    SEISMIC_NORMALIZED_STRIDE_0, SEISMIC_NORMALIZED_STRIDE_1));
                sum[r] = metal::fma(input, weight, sum[r]);
            }
        }
    }
    for (uint r = 0; r < 8; ++r) {
        ulong row = row_base + r;
        if (row >= SEISMIC_DIM_M) break;
        float projected = simd_sum(sum[r]);
        if (lane == 0) {
            if (kind == 0) activation_store(query_gate, row * query_width + output, projected);
            else if (kind == 1) activation_store(key, row * key_width + output, projected);
            else activation_store(value, row * key_width + output, projected);
        }
    }
}
