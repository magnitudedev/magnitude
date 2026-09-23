// The final projection is independent across rows and output channels.
inline ulong ro2(ulong a, ulong b, ulong sa, ulong sb) { return a * sa + b * sb; }
inline uint ro_code(device const uchar *bytes, ulong bit, uint width) {
    ulong byte = bit >> 3;
    uint shift = uint(bit & 7);
    uint value = uint(bytes[byte]) >> shift;
    if (shift + width > 8) value |= uint(bytes[byte + 1]) << (8 - shift);
    return value & ((1u << width) - 1u);
}
inline float ro_packet(device const uchar *base, ulong logical, uint kind, ulong size, ulong group,
    ulong words, ulong coefficients, ulong factor, ulong bias) {
    if (kind == 0) return *reinterpret_cast<device const float *>(base + logical * size);
    if (kind == 1) return float(*reinterpret_cast<device const half *>(base + logical * size));
    if (kind == 2) return as_type<float>(uint(*reinterpret_cast<device const ushort *>(base + logical * size)) << 16);
    device const uchar *resident = base + (logical / group) * size;
    ulong position = logical % group;
    if (kind == 8) return float(int(reinterpret_cast<device const char *>(resident + words)[position]))
        * float(*reinterpret_cast<device const half *>(resident + factor));
    if (kind == 14) {
        const int table[16] = {-127,-104,-83,-65,-49,-35,-22,-10,1,13,25,38,53,69,89,113};
        return reinterpret_cast<device const float *>(resident + factor)[position / 32]
            * float(table[ro_code(resident + words, position * 4, 4)]);
    }
    int code = int(ro_code(resident + words, position * kind, kind));
    if (kind == 6) code -= 32;
    ulong ci = position / (kind == 6 ? 16 : 32);
    if (kind == 6) return float(code * int(reinterpret_cast<device const char *>(resident + coefficients)[ci]))
        * float(*reinterpret_cast<device const half *>(resident + factor));
    uint scale_code = ro_code(resident + coefficients, ci * 12, 6);
    uint bias_code = ro_code(resident + coefficients, ci * 12 + 6, 6);
    return metal::fma(float(*reinterpret_cast<device const half *>(resident + factor)) * float(scale_code),
        float(code), -float(*reinterpret_cast<device const half *>(resident + bias)) * float(bias_code));
}
inline float ro_weight(device const uchar *base, ulong logical) {
#if defined(SEISMIC_OUTPUT_WEIGHT_REPRESENTATION_F32)
    return ro_packet(base, logical, 0, SEISMIC_OUTPUT_WEIGHT_PACKET_SIZE, 1, 0, 0, 0, 0);
#elif defined(SEISMIC_OUTPUT_WEIGHT_REPRESENTATION_F16)
    return ro_packet(base, logical, 1, SEISMIC_OUTPUT_WEIGHT_PACKET_SIZE, 1, 0, 0, 0, 0);
#elif defined(SEISMIC_OUTPUT_WEIGHT_REPRESENTATION_BF16)
    return ro_packet(base, logical, 2, SEISMIC_OUTPUT_WEIGHT_PACKET_SIZE, 1, 0, 0, 0, 0);
#elif defined(SEISMIC_OUTPUT_WEIGHT_REPRESENTATION_Q8G32S)
    return ro_packet(base, logical, 8, SEISMIC_OUTPUT_WEIGHT_PACKET_SIZE,
        SEISMIC_OUTPUT_WEIGHT_LOGICAL_GROUP, SEISMIC_OUTPUT_WEIGHT_PLANE_0_OFFSET, 0,
        SEISMIC_OUTPUT_WEIGHT_PLANE_1_OFFSET, 0);
#elif defined(SEISMIC_OUTPUT_WEIGHT_REPRESENTATION_Q4K)
    return ro_packet(base, logical, 4, SEISMIC_OUTPUT_WEIGHT_PACKET_SIZE,
        SEISMIC_OUTPUT_WEIGHT_LOGICAL_GROUP, SEISMIC_OUTPUT_WEIGHT_PLANE_0_OFFSET,
        SEISMIC_OUTPUT_WEIGHT_PLANE_1_OFFSET, SEISMIC_OUTPUT_WEIGHT_PLANE_2_OFFSET,
        SEISMIC_OUTPUT_WEIGHT_PLANE_3_OFFSET);
#elif defined(SEISMIC_OUTPUT_WEIGHT_REPRESENTATION_Q5K)
    return ro_packet(base, logical, 5, SEISMIC_OUTPUT_WEIGHT_PACKET_SIZE,
        SEISMIC_OUTPUT_WEIGHT_LOGICAL_GROUP, SEISMIC_OUTPUT_WEIGHT_PLANE_0_OFFSET,
        SEISMIC_OUTPUT_WEIGHT_PLANE_1_OFFSET, SEISMIC_OUTPUT_WEIGHT_PLANE_2_OFFSET,
        SEISMIC_OUTPUT_WEIGHT_PLANE_3_OFFSET);
#elif defined(SEISMIC_OUTPUT_WEIGHT_REPRESENTATION_Q6K)
    return ro_packet(base, logical, 6, SEISMIC_OUTPUT_WEIGHT_PACKET_SIZE,
        SEISMIC_OUTPUT_WEIGHT_LOGICAL_GROUP, SEISMIC_OUTPUT_WEIGHT_PLANE_0_OFFSET,
        SEISMIC_OUTPUT_WEIGHT_PLANE_1_OFFSET, SEISMIC_OUTPUT_WEIGHT_PLANE_2_OFFSET, 0);
#elif defined(SEISMIC_OUTPUT_WEIGHT_REPRESENTATION_IQ4G32)
    return ro_packet(base, logical, 14, SEISMIC_OUTPUT_WEIGHT_PACKET_SIZE,
        SEISMIC_OUTPUT_WEIGHT_LOGICAL_GROUP, SEISMIC_OUTPUT_WEIGHT_PLANE_0_OFFSET, 0,
        SEISMIC_OUTPUT_WEIGHT_PLANE_1_OFFSET, 0);
#else
#error "unsupported recurrent output weight"
#endif
}
inline float ro_activation(device const uchar *base, ulong logical) {
#if defined(SEISMIC_ELEMENT_A_REPRESENTATION_F32)
    return reinterpret_cast<device const float *>(base)[logical];
#elif defined(SEISMIC_ELEMENT_A_REPRESENTATION_F16)
    return float(reinterpret_cast<device const half *>(base)[logical]);
#elif defined(SEISMIC_ELEMENT_A_REPRESENTATION_BF16)
    return as_type<float>(uint(reinterpret_cast<device const ushort *>(base)[logical]) << 16);
#endif
}
inline float ro_round_activation(float value) {
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
kernel void qwen_recurrent_output(
    device const float *hidden [[buffer(SEISMIC_BUFFER_HIDDEN)]],
    device const uchar *gated [[buffer(SEISMIC_BUFFER_GATED)]],
    device const uchar *output_weight [[buffer(SEISMIC_BUFFER_OUTPUT_WEIGHT)]],
    device float *result [[buffer(SEISMIC_RESULT_0_BUFFER)]],
    constant ulong *seismic_words [[buffer(SEISMIC_BUFFER_WORDS)]],
    uint raw_index [[thread_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd_width [[threads_per_simdgroup]]) {
    const ulong row_tiles = (SEISMIC_DIM_M + 7) / 8;
    ulong dot = ulong(raw_index) / ulong(simd_width);
    if (dot >= row_tiles * SEISMIC_DIM_H) return;
    ulong output = dot / row_tiles;
    ulong row_base = (dot % row_tiles) * 8;
    ulong width = SEISMIC_DIM_G;
    float sums[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    for (ulong source = ulong(lane); source < width; source += ulong(simd_width)) {
        float weight = ro_weight(output_weight, output * width + source);
        for (uint r = 0; r < 8; ++r) {
            ulong row = row_base + r;
            if (row < SEISMIC_DIM_M) {
                float value = ro_activation(gated, ro2(row, source, SEISMIC_GATED_STRIDE_0,
                    SEISMIC_GATED_STRIDE_1));
                sums[r] = metal::fma(value, weight, sums[r]);
            }
        }
    }
    for (uint r = 0; r < 8; ++r) {
        ulong row = row_base + r;
        if (row >= SEISMIC_DIM_M) break;
        float projected = simd_sum(sums[r]);
        if (lane == 0)
            result[ro2(row, output, SEISMIC_RESULT_0_STRIDE_0, SEISMIC_RESULT_0_STRIDE_1)] =
                hidden[ro2(row, output, SEISMIC_HIDDEN_STRIDE_0, SEISMIC_HIDDEN_STRIDE_1)]
                + ro_round_activation(projected);
    }
}
