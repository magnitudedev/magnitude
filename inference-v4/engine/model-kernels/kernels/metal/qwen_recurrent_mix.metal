// Independent gating per row, value head, and recurrent component.
inline ulong rm2(ulong a, ulong b, ulong sa, ulong sb) { return a * sa + b * sb; }
inline ulong rm3(ulong a, ulong b, ulong c, ulong sa, ulong sb, ulong sc) {
    return a * sa + b * sb + c * sc;
}
inline float rm_norm(device const uchar *base, ulong logical) {
#if defined(SEISMIC_RECURRENT_NORM_REPRESENTATION_F32)
    return *reinterpret_cast<device const float *>(base + logical * SEISMIC_RECURRENT_NORM_PACKET_SIZE);
#elif defined(SEISMIC_RECURRENT_NORM_REPRESENTATION_F16)
    return float(*reinterpret_cast<device const half *>(base + logical * SEISMIC_RECURRENT_NORM_PACKET_SIZE));
#elif defined(SEISMIC_RECURRENT_NORM_REPRESENTATION_BF16)
    return as_type<float>(uint(*reinterpret_cast<device const ushort *>(base + logical * SEISMIC_RECURRENT_NORM_PACKET_SIZE)) << 16);
#else
#error "recurrent norm must be dense"
#endif
}
inline float rm_activation(device const uchar *base, ulong logical) {
#if defined(SEISMIC_ELEMENT_A_REPRESENTATION_F32)
    return reinterpret_cast<device const float *>(base)[logical];
#elif defined(SEISMIC_ELEMENT_A_REPRESENTATION_F16)
    return float(reinterpret_cast<device const half *>(base)[logical]);
#elif defined(SEISMIC_ELEMENT_A_REPRESENTATION_BF16)
    return as_type<float>(uint(reinterpret_cast<device const ushort *>(base)[logical]) << 16);
#endif
}
inline float rm_round_activation(float value) {
#if defined(SEISMIC_ELEMENT_A_REPRESENTATION_F32)
    return value;
#elif defined(SEISMIC_ELEMENT_A_REPRESENTATION_F16)
    return float(half(value));
#elif defined(SEISMIC_ELEMENT_A_REPRESENTATION_BF16)
    uint bits = as_type<uint>(value);
    bits += 0x7fffu + ((bits >> 16) & 1u);
    return as_type<float>(bits & 0xffff0000u);
#endif
}
inline void rm_store_activation(device uchar *base, ulong logical, float value) {
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
kernel void qwen_recurrent_mix(
    device const uchar *projection [[buffer(SEISMIC_BUFFER_PROJECTION)]],
    device const uchar *mixed [[buffer(SEISMIC_BUFFER_MIXED)]],
    device const uchar *recurrent_norm [[buffer(SEISMIC_BUFFER_RECURRENT_NORM)]],
    device uchar *gated [[buffer(SEISMIC_RESULT_0_BUFFER)]],
    constant ulong *seismic_words [[buffer(SEISMIC_BUFFER_WORDS)]],
    uint raw_index [[thread_position_in_grid]]) {
    ulong index = ulong(raw_index);
    if (index >= SEISMIC_DIM_M * SEISMIC_DIM_NV * SEISMIC_DIM_W) return;
    ulong row = index / (SEISMIC_DIM_NV * SEISMIC_DIM_W);
    ulong flat = index % (SEISMIC_DIM_NV * SEISMIC_DIM_W);
    ulong value_head = flat / SEISMIC_DIM_W;
    ulong column = flat % SEISMIC_DIM_W;
    float squares = 0.0f;
    for (ulong other = 0; other < SEISMIC_DIM_W; ++other) {
        float value = rm_activation(mixed, rm3(row, value_head, other,
            SEISMIC_MIXED_STRIDE_0, SEISMIC_MIXED_STRIDE_1, SEISMIC_MIXED_STRIDE_2));
        squares = metal::fma(value, value, squares);
    }
    float inverse = metal::rsqrt(squares / float(SEISMIC_DIM_W)
        + as_type<float>(uint(SEISMIC_PARAM_EPSILON)));
    ulong q_width = (2 * SEISMIC_DIM_NK + SEISMIC_DIM_NV) * SEISMIC_DIM_W;
    float gate = rm_activation(projection, rm2(row, q_width + flat,
        SEISMIC_PROJECTION_STRIDE_0, SEISMIC_PROJECTION_STRIDE_1));
    float normalized = rm_round_activation(rm_activation(mixed, rm3(row, value_head, column,
        SEISMIC_MIXED_STRIDE_0, SEISMIC_MIXED_STRIDE_1, SEISMIC_MIXED_STRIDE_2))
        * inverse * rm_norm(recurrent_norm, column * SEISMIC_RECURRENT_NORM_STRIDE_0));
    float activated = rm_round_activation(gate / (1.0f + metal::exp(-gate)));
    rm_store_activation(gated, rm2(row, flat, SEISMIC_RESULT_0_STRIDE_0,
        SEISMIC_RESULT_0_STRIDE_1), normalized * activated);
}
