inline ulong offset2(ulong row, ulong column, ulong stride0, ulong stride1) {
    return row * stride0 + column * stride1;
}

inline uint packed_code(device const uchar *bytes, ulong bit, uint width) {
    uint value = 0;
    for (uint offset = 0; offset < width; ++offset)
        value |= uint((bytes[(bit + offset) >> 3] >> ((bit + offset) & 7)) & 1) << offset;
    return value;
}
inline float resident_load(device const uchar *base, ulong logical, uint kind, ulong packet_size,
    ulong group, ulong words, ulong coefficients, ulong factor, ulong bias) {
    device const uchar *packet = base + (logical / group) * packet_size;
    ulong position = logical % group;
    if (kind == 8) return float(int(reinterpret_cast<device const char *>(packet + words)[position])) * float(*reinterpret_cast<device const half *>(packet + factor));
    if (kind == 14) {
        const int table[16] = {-127,-104,-83,-65,-49,-35,-22,-10,1,13,25,38,53,69,89,113};
        return reinterpret_cast<device const float *>(packet + factor)[position / 32] * float(table[packed_code(packet + words, position * 4, 4)]);
    }
    int code = int(packed_code(packet + words, position * kind, kind));
    if (kind == 6) code -= 32;
    ulong coefficient_index = position / (kind == 6 ? 16 : 32);
    if (kind == 6) return float(code * int(reinterpret_cast<device const char *>(packet + coefficients)[coefficient_index])) * float(*reinterpret_cast<device const half *>(packet + factor));
    uint scale_code = packed_code(packet + coefficients, coefficient_index * 12, 6);
    uint bias_code = packed_code(packet + coefficients, coefficient_index * 12 + 6, 6);
    return metal::fma(float(*reinterpret_cast<device const half *>(packet + factor)) * float(scale_code), float(code), -float(*reinterpret_cast<device const half *>(packet + bias)) * float(bias_code));
}

inline float load_table(device const uchar *base, ulong logical) {
#if defined(SEISMIC_TABLE_REPRESENTATION_F32)
    return *reinterpret_cast<device const float *>(base + logical * SEISMIC_TABLE_PACKET_SIZE);
#elif defined(SEISMIC_TABLE_REPRESENTATION_F16)
    return float(*reinterpret_cast<device const half *>(base + logical * SEISMIC_TABLE_PACKET_SIZE));
#elif defined(SEISMIC_TABLE_REPRESENTATION_BF16)
    ushort bits = *reinterpret_cast<device const ushort *>(base + logical * SEISMIC_TABLE_PACKET_SIZE);
    return as_type<float>(uint(bits) << 16);
#elif defined(SEISMIC_TABLE_REPRESENTATION_Q8G32S)
    return resident_load(base, logical, 8, SEISMIC_TABLE_PACKET_SIZE, SEISMIC_TABLE_LOGICAL_GROUP, SEISMIC_TABLE_PLANE_0_OFFSET, 0, SEISMIC_TABLE_PLANE_1_OFFSET, 0);
#elif defined(SEISMIC_TABLE_REPRESENTATION_Q4K)
    return resident_load(base, logical, 4, SEISMIC_TABLE_PACKET_SIZE, SEISMIC_TABLE_LOGICAL_GROUP, SEISMIC_TABLE_PLANE_0_OFFSET, SEISMIC_TABLE_PLANE_1_OFFSET, SEISMIC_TABLE_PLANE_2_OFFSET, SEISMIC_TABLE_PLANE_3_OFFSET);
#elif defined(SEISMIC_TABLE_REPRESENTATION_Q5K)
    return resident_load(base, logical, 5, SEISMIC_TABLE_PACKET_SIZE, SEISMIC_TABLE_LOGICAL_GROUP, SEISMIC_TABLE_PLANE_0_OFFSET, SEISMIC_TABLE_PLANE_1_OFFSET, SEISMIC_TABLE_PLANE_2_OFFSET, SEISMIC_TABLE_PLANE_3_OFFSET);
#elif defined(SEISMIC_TABLE_REPRESENTATION_Q6K)
    return resident_load(base, logical, 6, SEISMIC_TABLE_PACKET_SIZE, SEISMIC_TABLE_LOGICAL_GROUP, SEISMIC_TABLE_PLANE_0_OFFSET, SEISMIC_TABLE_PLANE_1_OFFSET, SEISMIC_TABLE_PLANE_2_OFFSET, 0);
#elif defined(SEISMIC_TABLE_REPRESENTATION_IQ4G32)
    return resident_load(base, logical, 14, SEISMIC_TABLE_PACKET_SIZE, SEISMIC_TABLE_LOGICAL_GROUP, SEISMIC_TABLE_PLANE_0_OFFSET, 0, SEISMIC_TABLE_PLANE_1_OFFSET, 0);
#else
#error "qwen_embedding_rows currently requires a dense table representation"
#endif
}

inline void store_activation(device uchar *base, ulong logical, float value) {
#if defined(SEISMIC_RESULT_0_REPRESENTATION_F32)
    *reinterpret_cast<device float *>(base + logical * SEISMIC_RESULT_0_PACKET_SIZE) = value;
#elif defined(SEISMIC_RESULT_0_REPRESENTATION_F16)
    *reinterpret_cast<device half *>(base + logical * SEISMIC_RESULT_0_PACKET_SIZE) = half(value);
#elif defined(SEISMIC_RESULT_0_REPRESENTATION_BF16)
    uint bits = as_type<uint>(value);
    uint rounded = bits + 0x7fffu + ((bits >> 16) & 1u);
    *reinterpret_cast<device ushort *>(base + logical * SEISMIC_RESULT_0_PACKET_SIZE) = ushort(rounded >> 16);
#else
#error "qwen_embedding_rows requires a dense activation representation"
#endif
}

kernel void qwen_embedding_rows(
    device const uchar *table [[buffer(SEISMIC_BUFFER_TABLE)]],
    device const int *tokens [[buffer(SEISMIC_BUFFER_TOKENS)]],
    device uchar *embedded [[buffer(SEISMIC_RESULT_0_BUFFER)]],
    device float *published [[buffer(SEISMIC_RESULT_1_BUFFER)]],
    constant ulong *seismic_words [[buffer(SEISMIC_BUFFER_WORDS)]],
    uint index [[thread_position_in_grid]])
{
    ulong count = SEISMIC_DIM_M * SEISMIC_DIM_D;
    if (ulong(index) >= count) return;
    ulong row = ulong(index) / SEISMIC_DIM_D;
    ulong column = ulong(index) % SEISMIC_DIM_D;
    int token = tokens[offset2(row, 0, SEISMIC_TOKENS_STRIDE_0, 0)];
    ulong table_index = ulong(token) * SEISMIC_DIM_D + column;
    float value = load_table(table, table_index);
    ulong activation_index = offset2(row, column, SEISMIC_RESULT_0_STRIDE_0, SEISMIC_RESULT_0_STRIDE_1);
    store_activation(embedded, activation_index, value);
    published[offset2(row, column, SEISMIC_RESULT_1_STRIDE_0, SEISMIC_RESULT_1_STRIDE_1)] = value;
}
