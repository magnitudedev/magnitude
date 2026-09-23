inline ulong offset2(ulong row, ulong column, ulong stride0, ulong stride1) { return row * stride0 + column * stride1; }
inline uint packed_code(device const uchar *bytes, ulong bit, uint width) {
    uint value = 0;
    for (uint offset = 0; offset < width; ++offset) value |= uint((bytes[(bit + offset) >> 3] >> ((bit + offset) & 7)) & 1) << offset;
    return value;
}
inline float resident_load(device const uchar *base, ulong logical, uint kind, ulong packet_size, ulong group, ulong words, ulong coefficients, ulong factor, ulong bias) {
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
inline float load_norm(device const uchar *base, ulong logical) {
#if defined(SEISMIC_NORM_REPRESENTATION_F32)
    return *reinterpret_cast<device const float *>(base + logical * SEISMIC_NORM_PACKET_SIZE);
#elif defined(SEISMIC_NORM_REPRESENTATION_F16)
    return float(*reinterpret_cast<device const half *>(base + logical * SEISMIC_NORM_PACKET_SIZE));
#elif defined(SEISMIC_NORM_REPRESENTATION_BF16)
    ushort bits = *reinterpret_cast<device const ushort *>(base + logical * SEISMIC_NORM_PACKET_SIZE);
    return as_type<float>(uint(bits) << 16);
#else
#error "qwen_selected_rows requires a dense norm representation"
#endif
}
inline float load_weight(device const uchar *base, ulong logical) {
#if defined(SEISMIC_WEIGHT_REPRESENTATION_F32)
    return *reinterpret_cast<device const float *>(base + logical * SEISMIC_WEIGHT_PACKET_SIZE);
#elif defined(SEISMIC_WEIGHT_REPRESENTATION_F16)
    return float(*reinterpret_cast<device const half *>(base + logical * SEISMIC_WEIGHT_PACKET_SIZE));
#elif defined(SEISMIC_WEIGHT_REPRESENTATION_BF16)
    ushort bits = *reinterpret_cast<device const ushort *>(base + logical * SEISMIC_WEIGHT_PACKET_SIZE);
    return as_type<float>(uint(bits) << 16);
#elif defined(SEISMIC_WEIGHT_REPRESENTATION_Q8G32S)
    return resident_load(base, logical, 8, SEISMIC_WEIGHT_PACKET_SIZE, SEISMIC_WEIGHT_LOGICAL_GROUP, SEISMIC_WEIGHT_PLANE_0_OFFSET, 0, SEISMIC_WEIGHT_PLANE_1_OFFSET, 0);
#elif defined(SEISMIC_WEIGHT_REPRESENTATION_Q4K)
    return resident_load(base, logical, 4, SEISMIC_WEIGHT_PACKET_SIZE, SEISMIC_WEIGHT_LOGICAL_GROUP, SEISMIC_WEIGHT_PLANE_0_OFFSET, SEISMIC_WEIGHT_PLANE_1_OFFSET, SEISMIC_WEIGHT_PLANE_2_OFFSET, SEISMIC_WEIGHT_PLANE_3_OFFSET);
#elif defined(SEISMIC_WEIGHT_REPRESENTATION_Q5K)
    return resident_load(base, logical, 5, SEISMIC_WEIGHT_PACKET_SIZE, SEISMIC_WEIGHT_LOGICAL_GROUP, SEISMIC_WEIGHT_PLANE_0_OFFSET, SEISMIC_WEIGHT_PLANE_1_OFFSET, SEISMIC_WEIGHT_PLANE_2_OFFSET, SEISMIC_WEIGHT_PLANE_3_OFFSET);
#elif defined(SEISMIC_WEIGHT_REPRESENTATION_Q6K)
    return resident_load(base, logical, 6, SEISMIC_WEIGHT_PACKET_SIZE, SEISMIC_WEIGHT_LOGICAL_GROUP, SEISMIC_WEIGHT_PLANE_0_OFFSET, SEISMIC_WEIGHT_PLANE_1_OFFSET, SEISMIC_WEIGHT_PLANE_2_OFFSET, 0);
#elif defined(SEISMIC_WEIGHT_REPRESENTATION_IQ4G32)
    return resident_load(base, logical, 14, SEISMIC_WEIGHT_PACKET_SIZE, SEISMIC_WEIGHT_LOGICAL_GROUP, SEISMIC_WEIGHT_PLANE_0_OFFSET, 0, SEISMIC_WEIGHT_PLANE_1_OFFSET, 0);
#else
#error "qwen_selected_rows currently requires dense output weights"
#endif
}
inline float round_feature(float value) {
#if defined(SEISMIC_ELEMENT_A_REPRESENTATION_F32)
    return value;
#elif defined(SEISMIC_ELEMENT_A_REPRESENTATION_F16)
    return float(half(value));
#elif defined(SEISMIC_ELEMENT_A_REPRESENTATION_BF16)
    uint bits = as_type<uint>(value);
    return as_type<float>((bits + 0x7fffu + ((bits >> 16) & 1u)) & 0xffff0000u);
#else
#error "qwen_selected_rows requires a dense activation representation"
#endif
}
inline void store_feature(device uchar *base, ulong logical, float value) {
#if defined(SEISMIC_RESULT_0_REPRESENTATION_F32)
    *reinterpret_cast<device float *>(base + logical * SEISMIC_RESULT_0_PACKET_SIZE) = value;
#elif defined(SEISMIC_RESULT_0_REPRESENTATION_F16)
    *reinterpret_cast<device half *>(base + logical * SEISMIC_RESULT_0_PACKET_SIZE) = half(value);
#elif defined(SEISMIC_RESULT_0_REPRESENTATION_BF16)
    *reinterpret_cast<device ushort *>(base + logical * SEISMIC_RESULT_0_PACKET_SIZE) = ushort(as_type<uint>(round_feature(value)) >> 16);
#endif
}
inline float inverse_rms(device const float *hidden, constant ulong *seismic_words, ulong source_row, float epsilon) {
    float squares = 0.0f;
    for (ulong column = 0; column < SEISMIC_DIM_D; ++column) {
        float value = hidden[offset2(source_row, column, SEISMIC_HIDDEN_STRIDE_0, SEISMIC_HIDDEN_STRIDE_1)];
        squares += value * value;
    }
    return metal::rsqrt(squares / float(SEISMIC_DIM_D) + epsilon);
}
kernel void qwen_selected_rows(
    device const float *hidden [[buffer(SEISMIC_BUFFER_HIDDEN)]],
    device const uchar *norm [[buffer(SEISMIC_BUFFER_NORM)]],
    device const uchar *weight [[buffer(SEISMIC_BUFFER_WEIGHT)]],
    device const int *out_rows [[buffer(SEISMIC_BUFFER_OUT_ROWS)]],
    device const int *selected [[buffer(SEISMIC_BUFFER_SELECTED)]],
    device uchar *features [[buffer(SEISMIC_RESULT_0_BUFFER)]],
    device float *logits [[buffer(SEISMIC_RESULT_1_BUFFER)]],
    constant ulong *seismic_words [[buffer(SEISMIC_BUFFER_WORDS)]],
    uint index [[thread_position_in_grid]]) {
    ulong width = SEISMIC_DIM_D + SEISMIC_DIM_SV;
    if (ulong(index) >= SEISMIC_DIM_O * width) return;
    ulong row = ulong(index) / width;
    ulong lane = ulong(index) % width;
    ulong source_row = ulong(out_rows[row * SEISMIC_OUT_ROWS_STRIDE_0]);
    float inverse = inverse_rms(hidden, seismic_words, source_row, as_type<float>(uint(SEISMIC_PARAM_EPSILON)));
    if (lane < SEISMIC_DIM_D) {
        float value = round_feature(hidden[offset2(source_row, lane, SEISMIC_HIDDEN_STRIDE_0, SEISMIC_HIDDEN_STRIDE_1)]
            * inverse * load_norm(norm, lane * SEISMIC_NORM_STRIDE_0));
        store_feature(features, offset2(row, lane, SEISMIC_RESULT_0_STRIDE_0, SEISMIC_RESULT_0_STRIDE_1), value);
    } else {
        ulong choice = lane - SEISMIC_DIM_D;
        ulong vocabulary = ulong(selected[choice * SEISMIC_SELECTED_STRIDE_0]);
        float sum = 0.0f;
        for (ulong source = 0; source < SEISMIC_DIM_D; ++source) {
            float value = round_feature(hidden[offset2(source_row, source, SEISMIC_HIDDEN_STRIDE_0, SEISMIC_HIDDEN_STRIDE_1)]
                * inverse * load_norm(norm, source * SEISMIC_NORM_STRIDE_0));
            sum += value * load_weight(weight, vocabulary * SEISMIC_DIM_D + source);
        }
        logits[offset2(row, choice, SEISMIC_RESULT_1_STRIDE_0, SEISMIC_RESULT_1_STRIDE_1)] = sum;
    }
}
