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
kernel void qwen_recurrent_normalize(
    device const float *hidden [[buffer(SEISMIC_BUFFER_HIDDEN)]],
    device const uchar *input_norm [[buffer(SEISMIC_BUFFER_INPUT_NORM)]],
    device uchar *normalized [[buffer(SEISMIC_RESULT_0_BUFFER)]],
    constant ulong *seismic_words [[buffer(SEISMIC_BUFFER_WORDS)]],
    uint row [[thread_position_in_grid]]) {
    if (ulong(row) >= SEISMIC_DIM_M) return;
    float squares = 0.0f;
    for (ulong column = 0; column < SEISMIC_DIM_H; ++column) {
        float value = hidden[at2(row, column, SEISMIC_HIDDEN_STRIDE_0, SEISMIC_HIDDEN_STRIDE_1)];
        squares = metal::fma(value, value, squares);
    }
    float inverse = metal::rsqrt(squares / float(SEISMIC_DIM_H)
        + as_type<float>(uint(SEISMIC_PARAM_EPSILON)));
    for (ulong column = 0; column < SEISMIC_DIM_H; ++column) {
        float value = hidden[at2(row, column, SEISMIC_HIDDEN_STRIDE_0, SEISMIC_HIDDEN_STRIDE_1)]
            * inverse * load_input_norm(input_norm, column * SEISMIC_INPUT_NORM_STRIDE_0);
        activation_store(normalized, at2(row, column,
            SEISMIC_RESULT_0_STRIDE_0, SEISMIC_RESULT_0_STRIDE_1), value);
    }
}
