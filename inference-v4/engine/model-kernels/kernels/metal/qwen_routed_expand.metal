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
inline float load_expert_gate(device const uchar *base, ulong logical) {
#if defined(SEISMIC_EXPERT_GATE_REPRESENTATION_F32)
    return XDENSE(SEISMIC_EXPERT_GATE, 0);
#elif defined(SEISMIC_EXPERT_GATE_REPRESENTATION_F16)
    return XDENSE(SEISMIC_EXPERT_GATE, 1);
#elif defined(SEISMIC_EXPERT_GATE_REPRESENTATION_BF16)
    return XDENSE(SEISMIC_EXPERT_GATE, 2);
#elif defined(SEISMIC_EXPERT_GATE_REPRESENTATION_Q8G32S)
    return XQ8(SEISMIC_EXPERT_GATE);
#elif defined(SEISMIC_EXPERT_GATE_REPRESENTATION_Q4K)
    return XQK(SEISMIC_EXPERT_GATE, 4);
#elif defined(SEISMIC_EXPERT_GATE_REPRESENTATION_Q5K)
    return XQK(SEISMIC_EXPERT_GATE, 5);
#elif defined(SEISMIC_EXPERT_GATE_REPRESENTATION_Q6K)
    return XQ6(SEISMIC_EXPERT_GATE);
#elif defined(SEISMIC_EXPERT_GATE_REPRESENTATION_IQ4G32)
    return XIQ(SEISMIC_EXPERT_GATE);
#else
#error "unsupported routed expert_gate representation"
#endif
}
inline float load_expert_up(device const uchar *base, ulong logical) {
#if defined(SEISMIC_EXPERT_UP_REPRESENTATION_F32)
    return XDENSE(SEISMIC_EXPERT_UP, 0);
#elif defined(SEISMIC_EXPERT_UP_REPRESENTATION_F16)
    return XDENSE(SEISMIC_EXPERT_UP, 1);
#elif defined(SEISMIC_EXPERT_UP_REPRESENTATION_BF16)
    return XDENSE(SEISMIC_EXPERT_UP, 2);
#elif defined(SEISMIC_EXPERT_UP_REPRESENTATION_Q8G32S)
    return XQ8(SEISMIC_EXPERT_UP);
#elif defined(SEISMIC_EXPERT_UP_REPRESENTATION_Q4K)
    return XQK(SEISMIC_EXPERT_UP, 4);
#elif defined(SEISMIC_EXPERT_UP_REPRESENTATION_Q5K)
    return XQK(SEISMIC_EXPERT_UP, 5);
#elif defined(SEISMIC_EXPERT_UP_REPRESENTATION_Q6K)
    return XQ6(SEISMIC_EXPERT_UP);
#elif defined(SEISMIC_EXPERT_UP_REPRESENTATION_IQ4G32)
    return XIQ(SEISMIC_EXPERT_UP);
#else
#error "unsupported routed expert_up representation"
#endif
}
inline float load_shared_gate(device const uchar *base, ulong logical) {
#if defined(SEISMIC_SHARED_GATE_REPRESENTATION_F32)
    return XDENSE(SEISMIC_SHARED_GATE, 0);
#elif defined(SEISMIC_SHARED_GATE_REPRESENTATION_F16)
    return XDENSE(SEISMIC_SHARED_GATE, 1);
#elif defined(SEISMIC_SHARED_GATE_REPRESENTATION_BF16)
    return XDENSE(SEISMIC_SHARED_GATE, 2);
#elif defined(SEISMIC_SHARED_GATE_REPRESENTATION_Q8G32S)
    return XQ8(SEISMIC_SHARED_GATE);
#elif defined(SEISMIC_SHARED_GATE_REPRESENTATION_Q4K)
    return XQK(SEISMIC_SHARED_GATE, 4);
#elif defined(SEISMIC_SHARED_GATE_REPRESENTATION_Q5K)
    return XQK(SEISMIC_SHARED_GATE, 5);
#elif defined(SEISMIC_SHARED_GATE_REPRESENTATION_Q6K)
    return XQ6(SEISMIC_SHARED_GATE);
#elif defined(SEISMIC_SHARED_GATE_REPRESENTATION_IQ4G32)
    return XIQ(SEISMIC_SHARED_GATE);
#else
#error "unsupported routed shared_gate representation"
#endif
}
inline float load_shared_up(device const uchar *base, ulong logical) {
#if defined(SEISMIC_SHARED_UP_REPRESENTATION_F32)
    return XDENSE(SEISMIC_SHARED_UP, 0);
#elif defined(SEISMIC_SHARED_UP_REPRESENTATION_F16)
    return XDENSE(SEISMIC_SHARED_UP, 1);
#elif defined(SEISMIC_SHARED_UP_REPRESENTATION_BF16)
    return XDENSE(SEISMIC_SHARED_UP, 2);
#elif defined(SEISMIC_SHARED_UP_REPRESENTATION_Q8G32S)
    return XQ8(SEISMIC_SHARED_UP);
#elif defined(SEISMIC_SHARED_UP_REPRESENTATION_Q4K)
    return XQK(SEISMIC_SHARED_UP, 4);
#elif defined(SEISMIC_SHARED_UP_REPRESENTATION_Q5K)
    return XQK(SEISMIC_SHARED_UP, 5);
#elif defined(SEISMIC_SHARED_UP_REPRESENTATION_Q6K)
    return XQ6(SEISMIC_SHARED_UP);
#elif defined(SEISMIC_SHARED_UP_REPRESENTATION_IQ4G32)
    return XIQ(SEISMIC_SHARED_UP);
#else
#error "unsupported routed shared_up representation"
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
kernel void qwen_routed_expand(
    device const uchar *normalized [[buffer(SEISMIC_BUFFER_NORMALIZED)]],
    device const uchar *expert_gate [[buffer(SEISMIC_BUFFER_EXPERT_GATE)]],
    device const uchar *expert_up [[buffer(SEISMIC_BUFFER_EXPERT_UP)]],
    device const uchar *shared_gate [[buffer(SEISMIC_BUFFER_SHARED_GATE)]],
    device const uchar *shared_up [[buffer(SEISMIC_BUFFER_SHARED_UP)]],
    device const float *shared_control [[buffer(SEISMIC_BUFFER_SHARED_CONTROL)]],
    device const int *routes [[buffer(SEISMIC_BUFFER_ROUTES)]],
    device uchar *expert_product [[buffer(SEISMIC_RESULT_0_BUFFER)]],
    device uchar *shared_product [[buffer(SEISMIC_RESULT_1_BUFFER)]],
    device uchar *shared_coefficient [[buffer(SEISMIC_RESULT_2_BUFFER)]],
    constant ulong *seismic_words [[buffer(SEISMIC_BUFFER_WORDS)]],
    uint index [[thread_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd_width [[threads_per_simdgroup]]) {
    ulong output_index = ulong(index) / ulong(simd_width);
    ulong per_row = SEISMIC_DIM_K * SEISMIC_DIM_F + SEISMIC_DIM_S + 1;
    if (output_index >= SEISMIC_DIM_O * per_row) return;
    ulong row = output_index / per_row;
    ulong item = output_index % per_row;

    if (item < SEISMIC_DIM_K * SEISMIC_DIM_F) {
        ulong choice = item / SEISMIC_DIM_F;
        ulong feature = item % SEISMIC_DIM_F;
        int expert_value = routes[row * SEISMIC_ROUTES_STRIDE_0
            + choice * SEISMIC_ROUTES_STRIDE_1];
        if (expert_value < 0 || ulong(expert_value) >= SEISMIC_DIM_E) return;
        ulong expert = ulong(expert_value);
        float gate = 0.0f;
        float up = 0.0f;
        for (ulong source = ulong(lane); source < SEISMIC_DIM_H; source += ulong(simd_width)) {
            float input = load_activation(normalized, row * SEISMIC_NORMALIZED_STRIDE_0
                + source * SEISMIC_NORMALIZED_STRIDE_1);
            ulong logical = (expert * SEISMIC_DIM_F + feature) * SEISMIC_DIM_H + source;
            gate = metal::fma(input, load_expert_gate(expert_gate, logical), gate);
            up = metal::fma(input, load_expert_up(expert_up, logical), up);
        }
        gate = round_activation(simd_sum(gate));
        up = round_activation(simd_sum(up));
        float activated = round_activation(gate / (1.0f + metal::exp(-gate)));
        float product = round_activation(activated * up);
        if (lane != 0) return;
        ulong logical = row * SEISMIC_RESULT_0_STRIDE_0
            + choice * SEISMIC_RESULT_0_STRIDE_1
            + feature * SEISMIC_RESULT_0_STRIDE_2;
        store_activation(expert_product, logical, product);
        return;
    }

    item -= SEISMIC_DIM_K * SEISMIC_DIM_F;
    if (item < SEISMIC_DIM_S) {
        ulong feature = item;
        float gate = 0.0f;
        float up = 0.0f;
        for (ulong source = ulong(lane); source < SEISMIC_DIM_H; source += ulong(simd_width)) {
            float input = load_activation(normalized, row * SEISMIC_NORMALIZED_STRIDE_0
                + source * SEISMIC_NORMALIZED_STRIDE_1);
            gate = metal::fma(input,
                load_shared_gate(shared_gate, feature * SEISMIC_DIM_H + source), gate);
            up = metal::fma(input,
                load_shared_up(shared_up, feature * SEISMIC_DIM_H + source), up);
        }
        gate = round_activation(simd_sum(gate));
        up = round_activation(simd_sum(up));
        float activated = round_activation(gate / (1.0f + metal::exp(-gate)));
        float product = round_activation(activated * up);
        if (lane != 0) return;
        store_activation(shared_product, row * SEISMIC_RESULT_1_STRIDE_0
            + feature * SEISMIC_RESULT_1_STRIDE_1, product);
        return;
    }

    float coefficient = 0.0f;
    for (ulong source = ulong(lane); source < SEISMIC_DIM_H; source += ulong(simd_width)) {
        float input = load_activation(normalized, row * SEISMIC_NORMALIZED_STRIDE_0
            + source * SEISMIC_NORMALIZED_STRIDE_1);
        coefficient = metal::fma(input,
            shared_control[source * SEISMIC_SHARED_CONTROL_STRIDE_0], coefficient);
    }
    coefficient = simd_sum(coefficient);
    if (lane != 0) return;
    coefficient = round_activation(1.0f / (1.0f + metal::exp(-coefficient)));
    store_activation(shared_coefficient, row * SEISMIC_RESULT_2_STRIDE_0, coefficient);
}
