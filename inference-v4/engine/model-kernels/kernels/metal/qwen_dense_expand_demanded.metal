inline ulong offset2(ulong row, ulong column, ulong stride0, ulong stride1) {
    return row * stride0 + column * stride1;
}
inline float load_f32(device const float *base, ulong logical) { return base[logical]; }
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
    if (kind == 8) {
        int code = int(reinterpret_cast<device const char *>(packet + words)[position]);
        return float(code) * float(*reinterpret_cast<device const half *>(packet + factor));
    }
    if (kind == 14) {
        const int table[16] = {-127,-104,-83,-65,-49,-35,-22,-10,1,13,25,38,53,69,89,113};
        uint code = packed_code(packet + words, position * 4, 4);
        float scale = reinterpret_cast<device const float *>(packet + factor)[position / 32];
        return scale * float(table[code]);
    }
    uint width = kind;
    int code = int(packed_code(packet + words, position * width, width));
    if (kind == 6) code -= 32;
    ulong coefficient_group = kind == 6 ? 16 : 32;
    ulong coefficient_index = position / coefficient_group;
    if (kind == 6) {
        int coefficient = int(reinterpret_cast<device const char *>(packet + coefficients)[coefficient_index]);
        return float(code * coefficient) * float(*reinterpret_cast<device const half *>(packet + factor));
    }
    uint scale_code = packed_code(packet + coefficients, (coefficient_index * 2) * 6, 6);
    uint bias_code = packed_code(packet + coefficients, (coefficient_index * 2 + 1) * 6, 6);
    float scale = float(*reinterpret_cast<device const half *>(packet + factor)) * float(scale_code);
    float offset = -float(*reinterpret_cast<device const half *>(packet + bias)) * float(bias_code);
    return metal::fma(scale, float(code), offset);
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
#error "qwen_dense_rows requires a dense norm representation"
#endif
}
inline float load_gate_weight(device const uchar *base, ulong logical) {
#if defined(SEISMIC_GATE_WEIGHT_REPRESENTATION_F32)
    return *reinterpret_cast<device const float *>(base + logical * SEISMIC_GATE_WEIGHT_PACKET_SIZE);
#elif defined(SEISMIC_GATE_WEIGHT_REPRESENTATION_F16)
    return float(*reinterpret_cast<device const half *>(base + logical * SEISMIC_GATE_WEIGHT_PACKET_SIZE));
#elif defined(SEISMIC_GATE_WEIGHT_REPRESENTATION_BF16)
    ushort bits = *reinterpret_cast<device const ushort *>(base + logical * SEISMIC_GATE_WEIGHT_PACKET_SIZE);
    return as_type<float>(uint(bits) << 16);
#elif defined(SEISMIC_GATE_WEIGHT_REPRESENTATION_Q8G32S)
    return resident_load(base, logical, 8, SEISMIC_GATE_WEIGHT_PACKET_SIZE, SEISMIC_GATE_WEIGHT_LOGICAL_GROUP, SEISMIC_GATE_WEIGHT_PLANE_0_OFFSET, 0, SEISMIC_GATE_WEIGHT_PLANE_1_OFFSET, 0);
#elif defined(SEISMIC_GATE_WEIGHT_REPRESENTATION_Q4K)
    return resident_load(base, logical, 4, SEISMIC_GATE_WEIGHT_PACKET_SIZE, SEISMIC_GATE_WEIGHT_LOGICAL_GROUP, SEISMIC_GATE_WEIGHT_PLANE_0_OFFSET, SEISMIC_GATE_WEIGHT_PLANE_1_OFFSET, SEISMIC_GATE_WEIGHT_PLANE_2_OFFSET, SEISMIC_GATE_WEIGHT_PLANE_3_OFFSET);
#elif defined(SEISMIC_GATE_WEIGHT_REPRESENTATION_Q5K)
    return resident_load(base, logical, 5, SEISMIC_GATE_WEIGHT_PACKET_SIZE, SEISMIC_GATE_WEIGHT_LOGICAL_GROUP, SEISMIC_GATE_WEIGHT_PLANE_0_OFFSET, SEISMIC_GATE_WEIGHT_PLANE_1_OFFSET, SEISMIC_GATE_WEIGHT_PLANE_2_OFFSET, SEISMIC_GATE_WEIGHT_PLANE_3_OFFSET);
#elif defined(SEISMIC_GATE_WEIGHT_REPRESENTATION_Q6K)
    return resident_load(base, logical, 6, SEISMIC_GATE_WEIGHT_PACKET_SIZE, SEISMIC_GATE_WEIGHT_LOGICAL_GROUP, SEISMIC_GATE_WEIGHT_PLANE_0_OFFSET, SEISMIC_GATE_WEIGHT_PLANE_1_OFFSET, SEISMIC_GATE_WEIGHT_PLANE_2_OFFSET, 0);
#elif defined(SEISMIC_GATE_WEIGHT_REPRESENTATION_IQ4G32)
    return resident_load(base, logical, 14, SEISMIC_GATE_WEIGHT_PACKET_SIZE, SEISMIC_GATE_WEIGHT_LOGICAL_GROUP, SEISMIC_GATE_WEIGHT_PLANE_0_OFFSET, 0, SEISMIC_GATE_WEIGHT_PLANE_1_OFFSET, 0);
#else
#error "unsupported qwen_dense_rows gate weight representation"
#endif
}
inline float load_up_weight(device const uchar *base, ulong logical) {
#if defined(SEISMIC_UP_WEIGHT_REPRESENTATION_F32)
    return *reinterpret_cast<device const float *>(base + logical * SEISMIC_UP_WEIGHT_PACKET_SIZE);
#elif defined(SEISMIC_UP_WEIGHT_REPRESENTATION_F16)
    return float(*reinterpret_cast<device const half *>(base + logical * SEISMIC_UP_WEIGHT_PACKET_SIZE));
#elif defined(SEISMIC_UP_WEIGHT_REPRESENTATION_BF16)
    ushort bits = *reinterpret_cast<device const ushort *>(base + logical * SEISMIC_UP_WEIGHT_PACKET_SIZE);
    return as_type<float>(uint(bits) << 16);
#elif defined(SEISMIC_UP_WEIGHT_REPRESENTATION_Q8G32S)
    return resident_load(base, logical, 8, SEISMIC_UP_WEIGHT_PACKET_SIZE, SEISMIC_UP_WEIGHT_LOGICAL_GROUP, SEISMIC_UP_WEIGHT_PLANE_0_OFFSET, 0, SEISMIC_UP_WEIGHT_PLANE_1_OFFSET, 0);
#elif defined(SEISMIC_UP_WEIGHT_REPRESENTATION_Q4K)
    return resident_load(base, logical, 4, SEISMIC_UP_WEIGHT_PACKET_SIZE, SEISMIC_UP_WEIGHT_LOGICAL_GROUP, SEISMIC_UP_WEIGHT_PLANE_0_OFFSET, SEISMIC_UP_WEIGHT_PLANE_1_OFFSET, SEISMIC_UP_WEIGHT_PLANE_2_OFFSET, SEISMIC_UP_WEIGHT_PLANE_3_OFFSET);
#elif defined(SEISMIC_UP_WEIGHT_REPRESENTATION_Q5K)
    return resident_load(base, logical, 5, SEISMIC_UP_WEIGHT_PACKET_SIZE, SEISMIC_UP_WEIGHT_LOGICAL_GROUP, SEISMIC_UP_WEIGHT_PLANE_0_OFFSET, SEISMIC_UP_WEIGHT_PLANE_1_OFFSET, SEISMIC_UP_WEIGHT_PLANE_2_OFFSET, SEISMIC_UP_WEIGHT_PLANE_3_OFFSET);
#elif defined(SEISMIC_UP_WEIGHT_REPRESENTATION_Q6K)
    return resident_load(base, logical, 6, SEISMIC_UP_WEIGHT_PACKET_SIZE, SEISMIC_UP_WEIGHT_LOGICAL_GROUP, SEISMIC_UP_WEIGHT_PLANE_0_OFFSET, SEISMIC_UP_WEIGHT_PLANE_1_OFFSET, SEISMIC_UP_WEIGHT_PLANE_2_OFFSET, 0);
#elif defined(SEISMIC_UP_WEIGHT_REPRESENTATION_IQ4G32)
    return resident_load(base, logical, 14, SEISMIC_UP_WEIGHT_PACKET_SIZE, SEISMIC_UP_WEIGHT_LOGICAL_GROUP, SEISMIC_UP_WEIGHT_PLANE_0_OFFSET, 0, SEISMIC_UP_WEIGHT_PLANE_1_OFFSET, 0);
#else
#error "unsupported qwen_dense_rows up weight representation"
#endif
}
inline float round_activation(float value) {
#if defined(SEISMIC_ELEMENT_A_REPRESENTATION_F32)
    return value;
#elif defined(SEISMIC_ELEMENT_A_REPRESENTATION_F16)
    return float(half(value));
#elif defined(SEISMIC_ELEMENT_A_REPRESENTATION_BF16)
    uint bits = as_type<uint>(value);
    return as_type<float>((bits + 0x7fffu + ((bits >> 16) & 1u)) & 0xffff0000u);
#else
#error "qwen_dense_rows requires a dense activation representation"
#endif
}
inline float normalized_at(device const float *residual, device const uchar *norm,
    constant ulong *seismic_words, ulong row, ulong column, float inverse) {
    return round_activation(load_f32(residual, offset2(row, column, SEISMIC_RESIDUAL_STRIDE_0, SEISMIC_RESIDUAL_STRIDE_1))
        * inverse * load_norm(norm, column * SEISMIC_NORM_STRIDE_0));
}
inline float activation_product(device const float *residual, device const uchar *norm,
    device const uchar *gate_weight, device const uchar *up_weight,
    constant ulong *seismic_words, ulong row, ulong feature, float inverse) {
    float gate = 0.0f;
    float up = 0.0f;
    for (ulong source = 0; source < SEISMIC_DIM_H; ++source) {
        float input = normalized_at(residual, norm, seismic_words, row, source, inverse);
        gate += input * load_gate_weight(gate_weight, feature * SEISMIC_DIM_H + source);
        up += input * load_up_weight(up_weight, feature * SEISMIC_DIM_H + source);
    }
    gate = round_activation(gate);
    up = round_activation(up);
    float activated = round_activation(gate / (1.0f + metal::exp(-gate)));
    return round_activation(activated * up);
}
kernel void qwen_dense_expand_demanded(
    device const float *residual [[buffer(SEISMIC_BUFFER_RESIDUAL)]],
    device const uchar *norm [[buffer(SEISMIC_BUFFER_NORM)]],
    device const uchar *gate_weight [[buffer(SEISMIC_BUFFER_GATE_WEIGHT)]],
    device const uchar *up_weight [[buffer(SEISMIC_BUFFER_UP_WEIGHT)]],
    device const int *out_rows [[buffer(SEISMIC_BUFFER_OUT_ROWS)]],
    device uchar *product [[buffer(SEISMIC_RESULT_0_BUFFER)]],
    constant ulong *seismic_words [[buffer(SEISMIC_BUFFER_WORDS)]],
    uint index [[thread_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd_width [[threads_per_simdgroup]]) {
    ulong output_index = ulong(index) / ulong(simd_width);
    if (output_index >= SEISMIC_DIM_O * SEISMIC_DIM_F) return;
    ulong row = output_index / SEISMIC_DIM_F;
    ulong source_row = ulong(out_rows[row * SEISMIC_OUT_ROWS_STRIDE_0]);
    ulong feature = output_index % SEISMIC_DIM_F;
    float squares = 0.0f;
    for (ulong source = ulong(lane); source < SEISMIC_DIM_H; source += ulong(simd_width)) {
        float value = load_f32(residual, offset2(source_row, source,
            SEISMIC_RESIDUAL_STRIDE_0, SEISMIC_RESIDUAL_STRIDE_1));
        squares = metal::fma(value, value, squares);
    }
    squares = simd_sum(squares);
    float inverse = metal::rsqrt(squares / float(SEISMIC_DIM_H)
        + as_type<float>(uint(SEISMIC_PARAM_EPS)));
    float gate = 0.0f;
    float up = 0.0f;
    for (ulong source = ulong(lane); source < SEISMIC_DIM_H; source += ulong(simd_width)) {
        float input = normalized_at(residual, norm, seismic_words, source_row, source, inverse);
        gate = metal::fma(input, load_gate_weight(gate_weight, feature * SEISMIC_DIM_H + source), gate);
        up = metal::fma(input, load_up_weight(up_weight, feature * SEISMIC_DIM_H + source), up);
    }
    gate = round_activation(simd_sum(gate));
    up = round_activation(simd_sum(up));
    float activated = round_activation(gate / (1.0f + metal::exp(-gate)));
    float value = round_activation(activated * up);
    if (lane != 0) return;
    ulong logical = row * SEISMIC_RESULT_0_STRIDE_0
        + feature * SEISMIC_RESULT_0_STRIDE_1;
#if defined(SEISMIC_ELEMENT_A_REPRESENTATION_F32)
    reinterpret_cast<device float *>(product)[logical] = value;
#elif defined(SEISMIC_ELEMENT_A_REPRESENTATION_F16)
    reinterpret_cast<device half *>(product)[logical] = half(value);
#elif defined(SEISMIC_ELEMENT_A_REPRESENTATION_BF16)
    uint bits = as_type<uint>(value); bits += 0x7fffu + ((bits >> 16) & 1u);
    reinterpret_cast<device ushort *>(product)[logical] = ushort(bits >> 16);
#endif
}
