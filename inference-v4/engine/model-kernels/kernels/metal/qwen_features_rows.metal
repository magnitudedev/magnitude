inline ulong offset2(ulong row, ulong column, ulong stride0, ulong stride1) { return row * stride0 + column * stride1; }
inline float load_norm(device const uchar *base, ulong logical) {
#if defined(SEISMIC_NORM_REPRESENTATION_F32)
    return *reinterpret_cast<device const float *>(base + logical * SEISMIC_NORM_PACKET_SIZE);
#elif defined(SEISMIC_NORM_REPRESENTATION_F16)
    return float(*reinterpret_cast<device const half *>(base + logical * SEISMIC_NORM_PACKET_SIZE));
#elif defined(SEISMIC_NORM_REPRESENTATION_BF16)
    ushort bits = *reinterpret_cast<device const ushort *>(base + logical * SEISMIC_NORM_PACKET_SIZE);
    return as_type<float>(uint(bits) << 16);
#else
#error "qwen_features_rows requires a dense norm representation"
#endif
}
inline void store_feature(device uchar *base, ulong logical, float value) {
#if defined(SEISMIC_RESULT_0_REPRESENTATION_F32)
    *reinterpret_cast<device float *>(base + logical * SEISMIC_RESULT_0_PACKET_SIZE) = value;
#elif defined(SEISMIC_RESULT_0_REPRESENTATION_F16)
    *reinterpret_cast<device half *>(base + logical * SEISMIC_RESULT_0_PACKET_SIZE) = half(value);
#elif defined(SEISMIC_RESULT_0_REPRESENTATION_BF16)
    uint bits = as_type<uint>(value);
    *reinterpret_cast<device ushort *>(base + logical * SEISMIC_RESULT_0_PACKET_SIZE) = ushort((bits + 0x7fffu + ((bits >> 16) & 1u)) >> 16);
#else
#error "qwen_features_rows requires a dense activation representation"
#endif
}
kernel void qwen_features_rows(
    device const float *hidden [[buffer(SEISMIC_BUFFER_HIDDEN)]],
    device const uchar *norm [[buffer(SEISMIC_BUFFER_NORM)]],
    device const int *out_rows [[buffer(SEISMIC_BUFFER_OUT_ROWS)]],
    device uchar *features [[buffer(SEISMIC_RESULT_0_BUFFER)]],
    constant ulong *seismic_words [[buffer(SEISMIC_BUFFER_WORDS)]],
    uint raw_row [[thread_position_in_grid]]) {
    ulong row = ulong(raw_row);
    if (row >= SEISMIC_DIM_O) return;
    ulong source_row = ulong(out_rows[row * SEISMIC_OUT_ROWS_STRIDE_0]);
    float squares = 0.0f;
    for (ulong source = 0; source < SEISMIC_DIM_D; ++source) {
        float value = hidden[offset2(source_row, source,
            SEISMIC_HIDDEN_STRIDE_0, SEISMIC_HIDDEN_STRIDE_1)];
        squares = metal::fma(value, value, squares);
    }
    float inverse = metal::rsqrt(squares / float(SEISMIC_DIM_D)
        + as_type<float>(uint(SEISMIC_PARAM_EPSILON)));
    for (ulong column = 0; column < SEISMIC_DIM_D; ++column) {
        float value = hidden[offset2(source_row, column,
            SEISMIC_HIDDEN_STRIDE_0, SEISMIC_HIDDEN_STRIDE_1)]
            * inverse * load_norm(norm, column * SEISMIC_NORM_STRIDE_0);
        store_feature(features, offset2(row, column,
            SEISMIC_RESULT_0_STRIDE_0, SEISMIC_RESULT_0_STRIDE_1), value);
    }
}
