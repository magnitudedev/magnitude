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
#define ATTENTION_WIDTH (2 * SEISMIC_DIM_P + SEISMIC_DIM_S)
inline float prepared_component(device const uchar *raw, device const float *norm,
    device const int *coordinates, device const int *components, ulong row, ulong head,
    ulong column, ulong head_count, float base, float epsilon, constant ulong *seismic_words) {
    float squares = 0.0f;
    ulong first = (row * head_count + head) * ATTENTION_WIDTH;
    for (ulong i = 0; i < ATTENTION_WIDTH; ++i) {
        float v = activation_load(raw, first + i);
        squares = metal::fma(v, v, squares);
    }
    float inverse = metal::rsqrt(squares / float(ATTENTION_WIDTH) + epsilon);
    float normalized = activation_load(raw, first + column) * inverse * norm[column];
    if (column >= 2 * SEISMIC_DIM_P) return normalized;
    ulong pair = column % SEISMIC_DIM_P;
    ulong paired = column < SEISMIC_DIM_P ? column + SEISMIC_DIM_P : column - SEISMIC_DIM_P;
    float other = activation_load(raw, first + paired) * inverse * norm[paired];
    int component = components[pair * SEISMIC_ROTARY_COMPONENTS_STRIDE_0];
    int coordinate = coordinates[at2(row, ulong(component),
        SEISMIC_COORDINATES_STRIDE_0, SEISMIC_COORDINATES_STRIDE_1)];
    float angle = float(coordinate) * metal::exp(-metal::log(base) * float(2 * pair)
        / float(2 * SEISMIC_DIM_P));
    return column < SEISMIC_DIM_P
        ? normalized * metal::cos(angle) - other * metal::sin(angle)
        : normalized * metal::cos(angle) + other * metal::sin(angle);
}
kernel void qwen_attention_prepare(
    device const uchar *query_gate [[buffer(SEISMIC_BUFFER_QUERY_GATE)]],
    device const uchar *key [[buffer(SEISMIC_BUFFER_KEY)]],
    device const float *query_norm [[buffer(SEISMIC_BUFFER_QUERY_NORM)]],
    device const float *key_norm [[buffer(SEISMIC_BUFFER_KEY_NORM)]],
    device const int *coordinates [[buffer(SEISMIC_BUFFER_COORDINATES)]],
    device const int *rotary_components [[buffer(SEISMIC_BUFFER_ROTARY_COMPONENTS)]],
    device uchar *query [[buffer(SEISMIC_RESULT_0_BUFFER)]],
    device uchar *prepared_key [[buffer(SEISMIC_RESULT_1_BUFFER)]],
    device uchar *gate [[buffer(SEISMIC_RESULT_2_BUFFER)]],
    constant ulong *seismic_words [[buffer(SEISMIC_BUFFER_WORDS)]],
    uint raw_index [[thread_position_in_grid]]) {
    ulong index = ulong(raw_index);
    ulong query_heads = SEISMIC_DIM_KV * SEISMIC_DIM_G;
    ulong query_count = SEISMIC_DIM_M * query_heads;
    float base = as_type<float>(uint(SEISMIC_PARAM_BASE));
    float epsilon = as_type<float>(uint(SEISMIC_PARAM_EPSILON));
    if (index < query_count) {
        ulong row = index / query_heads;
        ulong head = index % query_heads;
        ulong raw_query = (row * query_heads * 2 + head * 2) * ATTENTION_WIDTH;
        ulong raw_gate = raw_query + ATTENTION_WIDTH;
        for (ulong column = 0; column < ATTENTION_WIDTH; ++column) {
            float q = prepared_component(query_gate, query_norm, coordinates,
                rotary_components, row, head * 2, column, query_heads * 2,
                base, epsilon, seismic_words);
            activation_store(query, (row * query_heads + head) * ATTENTION_WIDTH + column, q);
            activation_store(gate, (row * query_heads + head) * ATTENTION_WIDTH + column,
                activation_load(query_gate, raw_gate + column));
        }
        return;
    }
    index -= query_count;
    ulong key_count = SEISMIC_DIM_M * SEISMIC_DIM_KV;
    if (index >= key_count) return;
    ulong row = index / SEISMIC_DIM_KV;
    ulong head = index % SEISMIC_DIM_KV;
    for (ulong column = 0; column < ATTENTION_WIDTH; ++column) {
        float k = prepared_component(key, key_norm, coordinates, rotary_components,
            row, head, column, SEISMIC_DIM_KV, base, epsilon, seismic_words);
        activation_store(prepared_key, (row * SEISMIC_DIM_KV + head) * ATTENTION_WIDTH + column, k);
    }
}
