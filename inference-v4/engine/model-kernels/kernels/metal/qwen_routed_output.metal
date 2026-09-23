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
inline float load_expert_down(device const uchar *base, ulong logical) {
#if defined(SEISMIC_EXPERT_DOWN_REPRESENTATION_F32)
    return XDENSE(SEISMIC_EXPERT_DOWN, 0);
#elif defined(SEISMIC_EXPERT_DOWN_REPRESENTATION_F16)
    return XDENSE(SEISMIC_EXPERT_DOWN, 1);
#elif defined(SEISMIC_EXPERT_DOWN_REPRESENTATION_BF16)
    return XDENSE(SEISMIC_EXPERT_DOWN, 2);
#elif defined(SEISMIC_EXPERT_DOWN_REPRESENTATION_Q8G32S)
    return XQ8(SEISMIC_EXPERT_DOWN);
#elif defined(SEISMIC_EXPERT_DOWN_REPRESENTATION_Q4K)
    return XQK(SEISMIC_EXPERT_DOWN, 4);
#elif defined(SEISMIC_EXPERT_DOWN_REPRESENTATION_Q5K)
    return XQK(SEISMIC_EXPERT_DOWN, 5);
#elif defined(SEISMIC_EXPERT_DOWN_REPRESENTATION_Q6K)
    return XQ6(SEISMIC_EXPERT_DOWN);
#elif defined(SEISMIC_EXPERT_DOWN_REPRESENTATION_IQ4G32)
    return XIQ(SEISMIC_EXPERT_DOWN);
#else
#error "unsupported routed expert_down representation"
#endif
}
inline float load_shared_down(device const uchar *base, ulong logical) {
#if defined(SEISMIC_SHARED_DOWN_REPRESENTATION_F32)
    return XDENSE(SEISMIC_SHARED_DOWN, 0);
#elif defined(SEISMIC_SHARED_DOWN_REPRESENTATION_F16)
    return XDENSE(SEISMIC_SHARED_DOWN, 1);
#elif defined(SEISMIC_SHARED_DOWN_REPRESENTATION_BF16)
    return XDENSE(SEISMIC_SHARED_DOWN, 2);
#elif defined(SEISMIC_SHARED_DOWN_REPRESENTATION_Q8G32S)
    return XQ8(SEISMIC_SHARED_DOWN);
#elif defined(SEISMIC_SHARED_DOWN_REPRESENTATION_Q4K)
    return XQK(SEISMIC_SHARED_DOWN, 4);
#elif defined(SEISMIC_SHARED_DOWN_REPRESENTATION_Q5K)
    return XQK(SEISMIC_SHARED_DOWN, 5);
#elif defined(SEISMIC_SHARED_DOWN_REPRESENTATION_Q6K)
    return XQ6(SEISMIC_SHARED_DOWN);
#elif defined(SEISMIC_SHARED_DOWN_REPRESENTATION_IQ4G32)
    return XIQ(SEISMIC_SHARED_DOWN);
#else
#error "unsupported routed shared_down representation"
#endif
}
inline float round_activation(float value) {
#if defined(SEISMIC_ELEMENT_A_REPRESENTATION_F32)
    return value;
#elif defined(SEISMIC_ELEMENT_A_REPRESENTATION_F16)
    return float(half(value));
#elif defined(SEISMIC_ELEMENT_A_REPRESENTATION_BF16)
    uint bits = as_type<uint>(value); return as_type<float>((bits + 0x7fffu + ((bits >> 16) & 1u)) & 0xffff0000u);
#else
#error "routed activation must be dense"
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
kernel void qwen_routed_output(
    device const float *residual [[buffer(SEISMIC_BUFFER_RESIDUAL)]],
    device const int *source_rows [[buffer(SEISMIC_BUFFER_SOURCE_ROWS)]],
    device const uchar *expert_product [[buffer(SEISMIC_BUFFER_EXPERT_PRODUCT)]],
    device const uchar *shared_product [[buffer(SEISMIC_BUFFER_SHARED_PRODUCT)]],
    device const uchar *shared_coefficient [[buffer(SEISMIC_BUFFER_SHARED_COEFFICIENT)]],
    device const uchar *expert_down [[buffer(SEISMIC_BUFFER_EXPERT_DOWN)]],
    device const uchar *shared_down [[buffer(SEISMIC_BUFFER_SHARED_DOWN)]],
    device const int *routes [[buffer(SEISMIC_BUFFER_ROUTES)]],
    device const float *scores [[buffer(SEISMIC_BUFFER_SCORES)]],
    device float *value [[buffer(SEISMIC_RESULT_0_BUFFER)]],
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

    float selected = 0.0f;
    for (ulong choice = 0; choice < SEISMIC_DIM_K; ++choice) {
        int expert_value = routes[row * SEISMIC_ROUTES_STRIDE_0
            + choice * SEISMIC_ROUTES_STRIDE_1];
        if (expert_value < 0 || ulong(expert_value) >= SEISMIC_DIM_E) return;
        ulong expert = ulong(expert_value);
        float projected = 0.0f;
        for (ulong feature = ulong(lane); feature < SEISMIC_DIM_F; feature += ulong(simd_width)) {
            float product = load_activation(expert_product,
                row * SEISMIC_EXPERT_PRODUCT_STRIDE_0
                + choice * SEISMIC_EXPERT_PRODUCT_STRIDE_1
                + feature * SEISMIC_EXPERT_PRODUCT_STRIDE_2);
            projected = metal::fma(product,
                load_expert_down(expert_down,
                    (expert * SEISMIC_DIM_H + column) * SEISMIC_DIM_F + feature),
                projected);
        }
        projected = round_activation(simd_sum(projected));
        if (lane == 0) {
            float score = scores[row * SEISMIC_SCORES_STRIDE_0
                + choice * SEISMIC_SCORES_STRIDE_1];
            selected = metal::fma(score, projected, selected);
        }
    }

    float shared = 0.0f;
    for (ulong feature = ulong(lane); feature < SEISMIC_DIM_S; feature += ulong(simd_width)) {
        float product = load_activation(shared_product,
            row * SEISMIC_SHARED_PRODUCT_STRIDE_0
            + feature * SEISMIC_SHARED_PRODUCT_STRIDE_1);
        shared = metal::fma(product,
            load_shared_down(shared_down, column * SEISMIC_DIM_S + feature), shared);
    }
    shared = round_activation(simd_sum(shared));
    if (lane != 0) return;
    float coefficient = load_activation(shared_coefficient,
        row * SEISMIC_SHARED_COEFFICIENT_STRIDE_0);
    value[row * SEISMIC_RESULT_0_STRIDE_0 + column * SEISMIC_RESULT_0_STRIDE_1] =
        residual[source_row * SEISMIC_RESIDUAL_STRIDE_0
            + column * SEISMIC_RESIDUAL_STRIDE_1]
        + selected + shared * coefficient;
}
