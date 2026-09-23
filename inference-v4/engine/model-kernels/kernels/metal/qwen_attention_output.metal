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
inline float attention_round(float value) {
#if defined(SEISMIC_ELEMENT_A_REPRESENTATION_F32)
    return value;
#elif defined(SEISMIC_ELEMENT_A_REPRESENTATION_F16)
    return float(half(value));
#elif defined(SEISMIC_ELEMENT_A_REPRESENTATION_BF16)
    uint bits = as_type<uint>(value); bits += 0x7fffu + ((bits >> 16) & 1u);
    return as_type<float>(bits & 0xffff0000u);
#endif
}
kernel void qwen_attention_output(
    device const float *hidden [[buffer(SEISMIC_BUFFER_HIDDEN)]],
    device const uchar *gated [[buffer(SEISMIC_BUFFER_GATED)]],
    device const uchar *prepared_key [[buffer(SEISMIC_BUFFER_PREPARED_KEY)]],
    device const uchar *value [[buffer(SEISMIC_BUFFER_VALUE)]],
    device const uchar *output_weight [[buffer(SEISMIC_BUFFER_OUTPUT_WEIGHT)]],
    device const int *destinations [[buffer(SEISMIC_BUFFER_DESTINATIONS)]],
    device uchar *history_key [[buffer(SEISMIC_BUFFER_HISTORY_KEY)]],
    device uchar *history_value [[buffer(SEISMIC_BUFFER_HISTORY_VALUE)]],
    device float *result [[buffer(SEISMIC_RESULT_0_BUFFER)]],
    constant ulong *seismic_words [[buffer(SEISMIC_BUFFER_WORDS)]],
    uint raw_index [[thread_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd_width [[threads_per_simdgroup]]) {
    const ulong row_tiles = (SEISMIC_DIM_M + 7) / 8;
    ulong output_count = row_tiles * SEISMIC_DIM_D;
    ulong output_groups = (output_count + 7) / 8;
    ulong output_threads = output_groups * 256;
    ulong raw = ulong(raw_index);
    if (raw < output_threads) {
        ulong dot = raw / ulong(simd_width);
        if (dot >= output_count) return;
        ulong column = dot / row_tiles;
        ulong row_base = (dot % row_tiles) * 8;
        ulong inner = SEISMIC_DIM_KV * SEISMIC_DIM_G * SEISMIC_DIM_W;
        float sums[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
        for (ulong source = ulong(lane); source < inner; source += ulong(simd_width)) {
            ulong head = source / SEISMIC_DIM_W;
            ulong component = source % SEISMIC_DIM_W;
            float weight = load_output_weight(output_weight, column * inner + source);
            for (uint r = 0; r < 8; ++r) {
                ulong row = row_base + r;
                if (row < SEISMIC_DIM_M) {
                    float input = activation_load(gated, at3(row, head, component,
                        SEISMIC_GATED_STRIDE_0, SEISMIC_GATED_STRIDE_1,
                        SEISMIC_GATED_STRIDE_2));
                    sums[r] = metal::fma(input, weight, sums[r]);
                }
            }
        }
        for (uint r = 0; r < 8; ++r) {
            ulong row = row_base + r;
            if (row >= SEISMIC_DIM_M) break;
            float projected = attention_round(simd_sum(sums[r]));
            if (lane == 0)
                result[at2(row, column, SEISMIC_RESULT_0_STRIDE_0,
                    SEISMIC_RESULT_0_STRIDE_1)] =
                    hidden[at2(row, column, SEISMIC_HIDDEN_STRIDE_0,
                        SEISMIC_HIDDEN_STRIDE_1)] + projected;
        }
        return;
    }
    ulong index = raw - output_threads;
    ulong state_count = SEISMIC_DIM_M * SEISMIC_DIM_KV * SEISMIC_DIM_W;
    if (index >= state_count) return;
    ulong row = index / (SEISMIC_DIM_KV * SEISMIC_DIM_W);
    ulong rem = index % (SEISMIC_DIM_KV * SEISMIC_DIM_W);
    ulong head = rem / SEISMIC_DIM_W;
    ulong column = rem % SEISMIC_DIM_W;
    int destination = destinations[row * SEISMIC_DESTINATIONS_STRIDE_0];
    if (destination < 0) return;
    float k = activation_load(prepared_key, at3(row, head, column,
        SEISMIC_PREPARED_KEY_STRIDE_0, SEISMIC_PREPARED_KEY_STRIDE_1,
        SEISMIC_PREPARED_KEY_STRIDE_2));
    float v = activation_load(value, row * SEISMIC_VALUE_STRIDE_0
        + (head * SEISMIC_DIM_W + column) * SEISMIC_VALUE_STRIDE_1);
    activation_store(history_key, at3(ulong(destination), head, column,
        SEISMIC_HISTORY_KEY_STRIDE_0, SEISMIC_HISTORY_KEY_STRIDE_1,
        SEISMIC_HISTORY_KEY_STRIDE_2), k);
    activation_store(history_value, at3(ulong(destination), head, column,
        SEISMIC_HISTORY_VALUE_STRIDE_0, SEISMIC_HISTORY_VALUE_STRIDE_1,
        SEISMIC_HISTORY_VALUE_STRIDE_2), v);
}
