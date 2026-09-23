inline ulong offset2(ulong row, ulong column, ulong stride0, ulong stride1) {
    return row * stride0 + column * stride1;
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
#error "unsupported routed norm representation"
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
#error "routed activation must be dense"
#endif
}
inline void store_activation(device uchar *base, ulong logical, float value) {
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
kernel void qwen_routed_normalize(
    device const float *residual [[buffer(SEISMIC_BUFFER_RESIDUAL)]],
    device const uchar *norm [[buffer(SEISMIC_BUFFER_NORM)]],
    device const int *source_rows [[buffer(SEISMIC_BUFFER_SOURCE_ROWS)]],
    device uchar *normalized [[buffer(SEISMIC_RESULT_0_BUFFER)]],
    constant ulong *seismic_words [[buffer(SEISMIC_BUFFER_WORDS)]],
    uint index [[thread_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd_width [[threads_per_simdgroup]]) {
    ulong output_index = ulong(index) / ulong(simd_width);
    if (output_index >= SEISMIC_DIM_O * SEISMIC_DIM_H) return;
    ulong row = output_index / SEISMIC_DIM_H;
    ulong column = output_index % SEISMIC_DIM_H;
    int source_value = source_rows[row * SEISMIC_SOURCE_ROWS_STRIDE_0];
    if (source_value < 0 || ulong(source_value) >= SEISMIC_DIM_M) return;
    ulong source_row = ulong(source_value);
    float squares = 0.0f;
    for (ulong source = ulong(lane); source < SEISMIC_DIM_H; source += ulong(simd_width)) {
        float value = residual[offset2(source_row, source,
            SEISMIC_RESIDUAL_STRIDE_0, SEISMIC_RESIDUAL_STRIDE_1)];
        squares = metal::fma(value, value, squares);
    }
    float inverse = metal::rsqrt(simd_sum(squares) / float(SEISMIC_DIM_H)
        + as_type<float>(uint(SEISMIC_PARAM_EPS)));
    if (lane != 0) return;
    float value = residual[offset2(source_row, column,
        SEISMIC_RESIDUAL_STRIDE_0, SEISMIC_RESIDUAL_STRIDE_1)];
    value = round_activation(value * inverse
        * load_norm(norm, column * SEISMIC_NORM_STRIDE_0));
    store_activation(normalized, offset2(row, column,
        SEISMIC_RESULT_0_STRIDE_0, SEISMIC_RESULT_0_STRIDE_1), value);
}
