inline ulong head_at2(ulong row, ulong column, ulong stride0, ulong stride1) {
    return row * stride0 + column * stride1;
}
inline uint head_code(device const uchar *bytes, ulong bit, uint width) {
    ulong byte = bit >> 3;
    uint shift = uint(bit & 7);
    uint value = uint(bytes[byte]) >> shift;
    if (shift + width > 8)
        value |= uint(bytes[byte + 1]) << (8 - shift);
    return value & ((1u << width) - 1u);
}
inline float head_resident(device const uchar *base, ulong logical, uint kind,
    ulong packet_size, ulong group, ulong words, ulong coefficients, ulong factor, ulong bias) {
    device const uchar *packet = base + (logical / group) * packet_size;
    ulong position = logical % group;
    if (kind == 8)
        return float(int(reinterpret_cast<device const char *>(packet + words)[position]))
            * float(*reinterpret_cast<device const half *>(packet + factor));
    if (kind == 14) {
        const int table[16] = {-127,-104,-83,-65,-49,-35,-22,-10,1,13,25,38,53,69,89,113};
        return reinterpret_cast<device const float *>(packet + factor)[position / 32]
            * float(table[head_code(packet + words, position * 4, 4)]);
    }
    int code = int(head_code(packet + words, position * kind, kind));
    if (kind == 6) code -= 32;
    ulong ci = position / (kind == 6 ? 16 : 32);
    if (kind == 6)
        return float(code * int(reinterpret_cast<device const char *>(packet + coefficients)[ci]))
            * float(*reinterpret_cast<device const half *>(packet + factor));
    uint scale_code = head_code(packet + coefficients, ci * 12, 6);
    uint bias_code = head_code(packet + coefficients, ci * 12 + 6, 6);
    return metal::fma(float(*reinterpret_cast<device const half *>(packet + factor))
        * float(scale_code), float(code),
        -float(*reinterpret_cast<device const half *>(packet + bias)) * float(bias_code));
}
inline float head_feature(device const uchar *base, ulong logical) {
#if defined(SEISMIC_ELEMENT_A_REPRESENTATION_F32)
    return *reinterpret_cast<device const float *>(base + logical * 4);
#elif defined(SEISMIC_ELEMENT_A_REPRESENTATION_F16)
    return float(*reinterpret_cast<device const half *>(base + logical * 2));
#elif defined(SEISMIC_ELEMENT_A_REPRESENTATION_BF16)
    return as_type<float>(uint(*reinterpret_cast<device const ushort *>(base + logical * 2)) << 16);
#else
#error "head logits require dense features"
#endif
}
inline float head_weight(device const uchar *base, ulong logical) {
#if defined(SEISMIC_WEIGHT_REPRESENTATION_F32)
    return *reinterpret_cast<device const float *>(base + logical * SEISMIC_WEIGHT_PACKET_SIZE);
#elif defined(SEISMIC_WEIGHT_REPRESENTATION_F16)
    return float(*reinterpret_cast<device const half *>(base + logical * SEISMIC_WEIGHT_PACKET_SIZE));
#elif defined(SEISMIC_WEIGHT_REPRESENTATION_BF16)
    return as_type<float>(uint(*reinterpret_cast<device const ushort *>(base + logical * SEISMIC_WEIGHT_PACKET_SIZE)) << 16);
#elif defined(SEISMIC_WEIGHT_REPRESENTATION_Q8G32S)
    return head_resident(base, logical, 8, SEISMIC_WEIGHT_PACKET_SIZE, SEISMIC_WEIGHT_LOGICAL_GROUP,
        SEISMIC_WEIGHT_PLANE_0_OFFSET, 0, SEISMIC_WEIGHT_PLANE_1_OFFSET, 0);
#elif defined(SEISMIC_WEIGHT_REPRESENTATION_Q4K)
    return head_resident(base, logical, 4, SEISMIC_WEIGHT_PACKET_SIZE, SEISMIC_WEIGHT_LOGICAL_GROUP,
        SEISMIC_WEIGHT_PLANE_0_OFFSET, SEISMIC_WEIGHT_PLANE_1_OFFSET,
        SEISMIC_WEIGHT_PLANE_2_OFFSET, SEISMIC_WEIGHT_PLANE_3_OFFSET);
#elif defined(SEISMIC_WEIGHT_REPRESENTATION_Q5K)
    return head_resident(base, logical, 5, SEISMIC_WEIGHT_PACKET_SIZE, SEISMIC_WEIGHT_LOGICAL_GROUP,
        SEISMIC_WEIGHT_PLANE_0_OFFSET, SEISMIC_WEIGHT_PLANE_1_OFFSET,
        SEISMIC_WEIGHT_PLANE_2_OFFSET, SEISMIC_WEIGHT_PLANE_3_OFFSET);
#elif defined(SEISMIC_WEIGHT_REPRESENTATION_Q6K)
    return head_resident(base, logical, 6, SEISMIC_WEIGHT_PACKET_SIZE, SEISMIC_WEIGHT_LOGICAL_GROUP,
        SEISMIC_WEIGHT_PLANE_0_OFFSET, SEISMIC_WEIGHT_PLANE_1_OFFSET,
        SEISMIC_WEIGHT_PLANE_2_OFFSET, 0);
#elif defined(SEISMIC_WEIGHT_REPRESENTATION_IQ4G32)
    return head_resident(base, logical, 14, SEISMIC_WEIGHT_PACKET_SIZE, SEISMIC_WEIGHT_LOGICAL_GROUP,
        SEISMIC_WEIGHT_PLANE_0_OFFSET, 0, SEISMIC_WEIGHT_PLANE_1_OFFSET, 0);
#else
#error "unsupported head logits weight"
#endif
}
kernel void head_logits_rows(
    device const uchar *features [[buffer(SEISMIC_BUFFER_FEATURES)]],
    device const uchar *weight [[buffer(SEISMIC_BUFFER_WEIGHT)]],
    device float *result [[buffer(SEISMIC_RESULT_0_BUFFER)]],
    constant ulong *seismic_words [[buffer(SEISMIC_BUFFER_WORDS)]],
    uint raw_index [[thread_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd_width [[threads_per_simdgroup]]) {
    ulong index = ulong(raw_index) / ulong(simd_width);
    if (index >= SEISMIC_DIM_O * SEISMIC_DIM_V) return;
    ulong row = index / SEISMIC_DIM_V;
    ulong output = index % SEISMIC_DIM_V;
    float sum = 0.0f;
#if defined(SEISMIC_WEIGHT_REPRESENTATION_Q4K) || defined(SEISMIC_WEIGHT_REPRESENTATION_Q5K)
    // One SIMD group owns one output row. Traverse resident packets once and
    // reuse each packet's affine metadata while lanes decode adjacent values.
#if defined(SEISMIC_WEIGHT_REPRESENTATION_Q4K)
    const uint code_bits = 4;
#else
    const uint code_bits = 5;
#endif
    ulong row_begin = output * SEISMIC_DIM_D;
    ulong row_end = row_begin + SEISMIC_DIM_D;
    ulong first_packet = row_begin / SEISMIC_WEIGHT_LOGICAL_GROUP;
    ulong last_packet = (row_end - 1) / SEISMIC_WEIGHT_LOGICAL_GROUP;
    for (ulong packet_index = first_packet; packet_index <= last_packet; ++packet_index) {
        device const uchar *packet = weight + packet_index * SEISMIC_WEIGHT_PACKET_SIZE;
        ulong packet_begin = packet_index * SEISMIC_WEIGHT_LOGICAL_GROUP;
        float factor = float(*reinterpret_cast<device const half *>(packet + SEISMIC_WEIGHT_PLANE_2_OFFSET));
        float bias = float(*reinterpret_cast<device const half *>(packet + SEISMIC_WEIGHT_PLANE_3_OFFSET));
        for (ulong group = 0; group < SEISMIC_WEIGHT_LOGICAL_GROUP; group += ulong(simd_width)) {
            ulong position = group + ulong(lane);
            ulong logical = packet_begin + position;
            if (position >= SEISMIC_WEIGHT_LOGICAL_GROUP || logical < row_begin || logical >= row_end) continue;
            ulong ci = position >> 5;
            uint scale_code = head_code(packet + SEISMIC_WEIGHT_PLANE_1_OFFSET, ci * 12, 6);
            uint bias_code = head_code(packet + SEISMIC_WEIGHT_PLANE_1_OFFSET, ci * 12 + 6, 6);
            uint code = head_code(packet + SEISMIC_WEIGHT_PLANE_0_OFFSET, position * code_bits, code_bits);
            float value = metal::fma(factor * float(scale_code), float(code), -bias * float(bias_code));
            sum = metal::fma(head_feature(features, head_at2(row, logical - row_begin,
                SEISMIC_FEATURES_STRIDE_0, SEISMIC_FEATURES_STRIDE_1)), value, sum);
        }
    }
#else
    for (ulong source = ulong(lane); source < SEISMIC_DIM_D; source += ulong(simd_width)) {
        sum = metal::fma(head_feature(features, head_at2(row, source,
            SEISMIC_FEATURES_STRIDE_0, SEISMIC_FEATURES_STRIDE_1)),
            head_weight(weight, output * SEISMIC_DIM_D + source), sum);
    }
#endif
    sum = simd_sum(sum);
    if (lane == 0)
        result[head_at2(row, output, SEISMIC_RESULT_0_STRIDE_0,
            SEISMIC_RESULT_0_STRIDE_1)] = sum;
}
