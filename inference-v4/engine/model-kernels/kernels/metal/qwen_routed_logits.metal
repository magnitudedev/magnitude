inline ulong offset2(ulong a, ulong b, ulong sa, ulong sb) { return a * sa + b * sb; }
inline ulong offset3(ulong a, ulong b, ulong c, ulong sa, ulong sb, ulong sc) {
    return a * sa + b * sb + c * sc;
}
inline uint packed_code(device const uchar *bytes, ulong bit, uint width) {
    uint value = 0;
    for (uint offset = 0; offset < width; ++offset) value |= uint((bytes[(bit + offset) >> 3] >> ((bit + offset) & 7)) & 1) << offset;
    return value;
}
inline float load_packet(device const uchar *base, ulong logical, uint kind, ulong packet_size,
    ulong group, ulong words, ulong coefficients, ulong factor, ulong bias) {
    if (kind == 0) return *reinterpret_cast<device const float *>(base + logical * packet_size);
    if (kind == 1) return float(*reinterpret_cast<device const half *>(base + logical * packet_size));
    if (kind == 2) return as_type<float>(uint(*reinterpret_cast<device const ushort *>(base + logical * packet_size)) << 16);
    device const uchar *packet = base + (logical / group) * packet_size;
    ulong position = logical % group;
    if (kind == 8) return float(int(reinterpret_cast<device const char *>(packet + words)[position])) * float(*reinterpret_cast<device const half *>(packet + factor));
    if (kind == 14) {
        const int table[16] = {-127,-104,-83,-65,-49,-35,-22,-10,1,13,25,38,53,69,89,113};
        return reinterpret_cast<device const float *>(packet + factor)[position / 32] * float(table[packed_code(packet + words, position * 4, 4)]);
    }
    int code = int(packed_code(packet + words, position * kind, kind));
    if (kind == 6) code -= 32;
    ulong ci = position / (kind == 6 ? 16 : 32);
    if (kind == 6) return float(code * int(reinterpret_cast<device const char *>(packet + coefficients)[ci])) * float(*reinterpret_cast<device const half *>(packet + factor));
    uint scale_code = packed_code(packet + coefficients, ci * 12, 6);
    uint bias_code = packed_code(packet + coefficients, ci * 12 + 6, 6);
    return metal::fma(float(*reinterpret_cast<device const half *>(packet + factor)) * float(scale_code), float(code), -float(*reinterpret_cast<device const half *>(packet + bias)) * float(bias_code));
}
#define XDENSE(PREFIX, KIND) load_packet(base, logical, KIND, PREFIX##_PACKET_SIZE, 1, 0, 0, 0, 0)
#define XQ8(PREFIX) load_packet(base, logical, 8, PREFIX##_PACKET_SIZE, PREFIX##_LOGICAL_GROUP, PREFIX##_PLANE_0_OFFSET, 0, PREFIX##_PLANE_1_OFFSET, 0)
#define XQK(PREFIX, KIND) load_packet(base, logical, KIND, PREFIX##_PACKET_SIZE, PREFIX##_LOGICAL_GROUP, PREFIX##_PLANE_0_OFFSET, PREFIX##_PLANE_1_OFFSET, PREFIX##_PLANE_2_OFFSET, PREFIX##_PLANE_3_OFFSET)
#define XQ6(PREFIX) load_packet(base, logical, 6, PREFIX##_PACKET_SIZE, PREFIX##_LOGICAL_GROUP, PREFIX##_PLANE_0_OFFSET, PREFIX##_PLANE_1_OFFSET, PREFIX##_PLANE_2_OFFSET, 0)
#define XIQ(PREFIX) load_packet(base, logical, 14, PREFIX##_PACKET_SIZE, PREFIX##_LOGICAL_GROUP, PREFIX##_PLANE_0_OFFSET, 0, PREFIX##_PLANE_1_OFFSET, 0)
inline float load_router(device const uchar *base, ulong logical) {
#if defined(SEISMIC_ROUTER_WEIGHT_REPRESENTATION_F32)
    return XDENSE(SEISMIC_ROUTER_WEIGHT, 0);
#elif defined(SEISMIC_ROUTER_WEIGHT_REPRESENTATION_F16)
    return XDENSE(SEISMIC_ROUTER_WEIGHT, 1);
#elif defined(SEISMIC_ROUTER_WEIGHT_REPRESENTATION_BF16)
    return XDENSE(SEISMIC_ROUTER_WEIGHT, 2);
#elif defined(SEISMIC_ROUTER_WEIGHT_REPRESENTATION_Q8G32S)
    return XQ8(SEISMIC_ROUTER_WEIGHT);
#elif defined(SEISMIC_ROUTER_WEIGHT_REPRESENTATION_Q4K)
    return XQK(SEISMIC_ROUTER_WEIGHT, 4);
#elif defined(SEISMIC_ROUTER_WEIGHT_REPRESENTATION_Q5K)
    return XQK(SEISMIC_ROUTER_WEIGHT, 5);
#elif defined(SEISMIC_ROUTER_WEIGHT_REPRESENTATION_Q6K)
    return XQ6(SEISMIC_ROUTER_WEIGHT);
#elif defined(SEISMIC_ROUTER_WEIGHT_REPRESENTATION_IQ4G32)
    return XIQ(SEISMIC_ROUTER_WEIGHT);
#else
#error "unsupported routed router representation"
#endif
}

inline float load_activation(device const uchar *base, ulong logical) {
#if defined(SEISMIC_ELEMENT_A_REPRESENTATION_F32)
    return reinterpret_cast<device const float *>(base)[logical];
#elif defined(SEISMIC_ELEMENT_A_REPRESENTATION_F16)
    return float(reinterpret_cast<device const half *>(base)[logical]);
#elif defined(SEISMIC_ELEMENT_A_REPRESENTATION_BF16)
    return as_type<float>(uint(reinterpret_cast<device const ushort *>(base)[logical]) << 16);
#else
#error "routed activation must be dense"
#endif
}
kernel void qwen_routed_logits(
    device const uchar *normalized [[buffer(SEISMIC_BUFFER_NORMALIZED)]],
    device const uchar *router_weight [[buffer(SEISMIC_BUFFER_ROUTER_WEIGHT)]],
    device float *logits [[buffer(SEISMIC_RESULT_0_BUFFER)]],
    constant ulong *seismic_words [[buffer(SEISMIC_BUFFER_WORDS)]],
    uint index [[thread_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd_width [[threads_per_simdgroup]]) {
    ulong output_index = ulong(index) / ulong(simd_width);
    if (output_index >= SEISMIC_DIM_O * SEISMIC_DIM_E) return;
    ulong row = output_index / SEISMIC_DIM_E;
    ulong expert = output_index % SEISMIC_DIM_E;
    float sum = 0.0f;
    for (ulong source = ulong(lane); source < SEISMIC_DIM_H; source += ulong(simd_width)) {
        float input = load_activation(normalized, row * SEISMIC_NORMALIZED_STRIDE_0
            + source * SEISMIC_NORMALIZED_STRIDE_1);
        sum = metal::fma(input,
            load_router(router_weight, expert * SEISMIC_DIM_H + source), sum);
    }
    sum = simd_sum(sum);
    if (lane != 0) return;
    logits[row * SEISMIC_RESULT_0_STRIDE_0 + expert * SEISMIC_RESULT_0_STRIDE_1] = sum;
}
