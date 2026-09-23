// One independent projection per row and output coordinate. Quantized weights
// are decoded from their admitted resident packets, without materializing a
// dense copy or repeating projection work during the recurrent scan.
inline ulong rp2(ulong a, ulong b, ulong sa, ulong sb) { return a * sa + b * sb; }
inline uint rp_code(device const uchar *bytes, ulong bit, uint width) {
    ulong byte = bit >> 3;
    uint shift = uint(bit & 7);
    uint value = uint(bytes[byte]) >> shift;
    if (shift + width > 8) value |= uint(bytes[byte + 1]) << (8 - shift);
    return value & ((1u << width) - 1u);
}
inline float rp_packet(device const uchar *base, ulong logical, uint kind, ulong size, ulong group,
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
            * float(table[rp_code(resident + words, position * 4, 4)]);
    }
    int code = int(rp_code(resident + words, position * kind, kind));
    if (kind == 6) code -= 32;
    ulong ci = position / (kind == 6 ? 16 : 32);
    if (kind == 6) return float(code * int(reinterpret_cast<device const char *>(resident + coefficients)[ci]))
        * float(*reinterpret_cast<device const half *>(resident + factor));
    uint scale_code = rp_code(resident + coefficients, ci * 12, 6);
    uint bias_code = rp_code(resident + coefficients, ci * 12 + 6, 6);
    return metal::fma(float(*reinterpret_cast<device const half *>(resident + factor)) * float(scale_code),
        float(code), -float(*reinterpret_cast<device const half *>(resident + bias)) * float(bias_code));
}
#define RP_DENSE(PREFIX, KIND) rp_packet(base, logical, KIND, PREFIX##_PACKET_SIZE, 1, 0, 0, 0, 0)
#define RP_Q8(PREFIX) rp_packet(base, logical, 8, PREFIX##_PACKET_SIZE, PREFIX##_LOGICAL_GROUP, PREFIX##_PLANE_0_OFFSET, 0, PREFIX##_PLANE_1_OFFSET, 0)
#define RP_QK(PREFIX, KIND) rp_packet(base, logical, KIND, PREFIX##_PACKET_SIZE, PREFIX##_LOGICAL_GROUP, PREFIX##_PLANE_0_OFFSET, PREFIX##_PLANE_1_OFFSET, PREFIX##_PLANE_2_OFFSET, PREFIX##_PLANE_3_OFFSET)
#define RP_Q6(PREFIX) rp_packet(base, logical, 6, PREFIX##_PACKET_SIZE, PREFIX##_LOGICAL_GROUP, PREFIX##_PLANE_0_OFFSET, PREFIX##_PLANE_1_OFFSET, PREFIX##_PLANE_2_OFFSET, 0)
#define RP_IQ(PREFIX) rp_packet(base, logical, 14, PREFIX##_PACKET_SIZE, PREFIX##_LOGICAL_GROUP, PREFIX##_PLANE_0_OFFSET, 0, PREFIX##_PLANE_1_OFFSET, 0)
inline float rp_qkv(device const uchar *base, ulong logical) {
#if defined(SEISMIC_QKV_WEIGHT_REPRESENTATION_F32)
    return RP_DENSE(SEISMIC_QKV_WEIGHT, 0);
#elif defined(SEISMIC_QKV_WEIGHT_REPRESENTATION_F16)
    return RP_DENSE(SEISMIC_QKV_WEIGHT, 1);
#elif defined(SEISMIC_QKV_WEIGHT_REPRESENTATION_BF16)
    return RP_DENSE(SEISMIC_QKV_WEIGHT, 2);
#elif defined(SEISMIC_QKV_WEIGHT_REPRESENTATION_Q8G32S)
    return RP_Q8(SEISMIC_QKV_WEIGHT);
#elif defined(SEISMIC_QKV_WEIGHT_REPRESENTATION_Q4K)
    return RP_QK(SEISMIC_QKV_WEIGHT, 4);
#elif defined(SEISMIC_QKV_WEIGHT_REPRESENTATION_Q5K)
    return RP_QK(SEISMIC_QKV_WEIGHT, 5);
#elif defined(SEISMIC_QKV_WEIGHT_REPRESENTATION_Q6K)
    return RP_Q6(SEISMIC_QKV_WEIGHT);
#elif defined(SEISMIC_QKV_WEIGHT_REPRESENTATION_IQ4G32)
    return RP_IQ(SEISMIC_QKV_WEIGHT);
#else
#error "unsupported recurrent qkv weight"
#endif
}
inline float rp_gate(device const uchar *base, ulong logical) {
#if defined(SEISMIC_GATE_WEIGHT_REPRESENTATION_F32)
    return RP_DENSE(SEISMIC_GATE_WEIGHT, 0);
#elif defined(SEISMIC_GATE_WEIGHT_REPRESENTATION_F16)
    return RP_DENSE(SEISMIC_GATE_WEIGHT, 1);
#elif defined(SEISMIC_GATE_WEIGHT_REPRESENTATION_BF16)
    return RP_DENSE(SEISMIC_GATE_WEIGHT, 2);
#elif defined(SEISMIC_GATE_WEIGHT_REPRESENTATION_Q8G32S)
    return RP_Q8(SEISMIC_GATE_WEIGHT);
#elif defined(SEISMIC_GATE_WEIGHT_REPRESENTATION_Q4K)
    return RP_QK(SEISMIC_GATE_WEIGHT, 4);
#elif defined(SEISMIC_GATE_WEIGHT_REPRESENTATION_Q5K)
    return RP_QK(SEISMIC_GATE_WEIGHT, 5);
#elif defined(SEISMIC_GATE_WEIGHT_REPRESENTATION_Q6K)
    return RP_Q6(SEISMIC_GATE_WEIGHT);
#elif defined(SEISMIC_GATE_WEIGHT_REPRESENTATION_IQ4G32)
    return RP_IQ(SEISMIC_GATE_WEIGHT);
#else
#error "unsupported recurrent gate weight"
#endif
}
inline float rp_alpha(device const uchar *base, ulong logical) {
#if defined(SEISMIC_ALPHA_WEIGHT_REPRESENTATION_F32)
    return RP_DENSE(SEISMIC_ALPHA_WEIGHT, 0);
#elif defined(SEISMIC_ALPHA_WEIGHT_REPRESENTATION_F16)
    return RP_DENSE(SEISMIC_ALPHA_WEIGHT, 1);
#elif defined(SEISMIC_ALPHA_WEIGHT_REPRESENTATION_BF16)
    return RP_DENSE(SEISMIC_ALPHA_WEIGHT, 2);
#elif defined(SEISMIC_ALPHA_WEIGHT_REPRESENTATION_Q8G32S)
    return RP_Q8(SEISMIC_ALPHA_WEIGHT);
#elif defined(SEISMIC_ALPHA_WEIGHT_REPRESENTATION_Q4K)
    return RP_QK(SEISMIC_ALPHA_WEIGHT, 4);
#elif defined(SEISMIC_ALPHA_WEIGHT_REPRESENTATION_Q5K)
    return RP_QK(SEISMIC_ALPHA_WEIGHT, 5);
#elif defined(SEISMIC_ALPHA_WEIGHT_REPRESENTATION_Q6K)
    return RP_Q6(SEISMIC_ALPHA_WEIGHT);
#elif defined(SEISMIC_ALPHA_WEIGHT_REPRESENTATION_IQ4G32)
    return RP_IQ(SEISMIC_ALPHA_WEIGHT);
#else
#error "unsupported recurrent alpha weight"
#endif
}
inline float rp_beta(device const uchar *base, ulong logical) {
#if defined(SEISMIC_BETA_WEIGHT_REPRESENTATION_F32)
    return RP_DENSE(SEISMIC_BETA_WEIGHT, 0);
#elif defined(SEISMIC_BETA_WEIGHT_REPRESENTATION_F16)
    return RP_DENSE(SEISMIC_BETA_WEIGHT, 1);
#elif defined(SEISMIC_BETA_WEIGHT_REPRESENTATION_BF16)
    return RP_DENSE(SEISMIC_BETA_WEIGHT, 2);
#elif defined(SEISMIC_BETA_WEIGHT_REPRESENTATION_Q8G32S)
    return RP_Q8(SEISMIC_BETA_WEIGHT);
#elif defined(SEISMIC_BETA_WEIGHT_REPRESENTATION_Q4K)
    return RP_QK(SEISMIC_BETA_WEIGHT, 4);
#elif defined(SEISMIC_BETA_WEIGHT_REPRESENTATION_Q5K)
    return RP_QK(SEISMIC_BETA_WEIGHT, 5);
#elif defined(SEISMIC_BETA_WEIGHT_REPRESENTATION_Q6K)
    return RP_Q6(SEISMIC_BETA_WEIGHT);
#elif defined(SEISMIC_BETA_WEIGHT_REPRESENTATION_IQ4G32)
    return RP_IQ(SEISMIC_BETA_WEIGHT);
#else
#error "unsupported recurrent beta weight"
#endif
}
inline float rp_activation(device const uchar *base, ulong logical) {
#if defined(SEISMIC_ELEMENT_A_REPRESENTATION_F32)
    return reinterpret_cast<device const float *>(base)[logical];
#elif defined(SEISMIC_ELEMENT_A_REPRESENTATION_F16)
    return float(reinterpret_cast<device const half *>(base)[logical]);
#elif defined(SEISMIC_ELEMENT_A_REPRESENTATION_BF16)
    return as_type<float>(uint(reinterpret_cast<device const ushort *>(base)[logical]) << 16);
#endif
}
inline void rp_store_activation(device uchar *base, ulong logical, float value) {
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
kernel void qwen_recurrent_project(
    device const uchar *normalized [[buffer(SEISMIC_BUFFER_NORMALIZED)]],
    device const uchar *qkv_weight [[buffer(SEISMIC_BUFFER_QKV_WEIGHT)]],
    device const uchar *gate_weight [[buffer(SEISMIC_BUFFER_GATE_WEIGHT)]],
    device const uchar *alpha_weight [[buffer(SEISMIC_BUFFER_ALPHA_WEIGHT)]],
    device const uchar *beta_weight [[buffer(SEISMIC_BUFFER_BETA_WEIGHT)]],
    device uchar *projection [[buffer(SEISMIC_RESULT_0_BUFFER)]],
    constant ulong *seismic_words [[buffer(SEISMIC_BUFFER_WORDS)]],
    uint raw_index [[thread_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd_width [[threads_per_simdgroup]]) {
    ulong q_width = (2 * SEISMIC_DIM_NK + SEISMIC_DIM_NV) * SEISMIC_DIM_W;
    ulong gate_width = SEISMIC_DIM_NV * SEISMIC_DIM_W;
    ulong stride = q_width + gate_width + 2 * SEISMIC_DIM_NV;
    const ulong row_tiles = (SEISMIC_DIM_M + 7) / 8;
    ulong dot = ulong(raw_index) / ulong(simd_width);
    if (dot >= row_tiles * stride) return;
    ulong output = dot / row_tiles;
    ulong row_base = (dot % row_tiles) * 8;
    uint kind = output < q_width ? 0 : (output < q_width + gate_width ? 1
        : (output < q_width + gate_width + SEISMIC_DIM_NV ? 2 : 3));
    ulong weight_row = kind == 0 ? output : (kind == 1 ? output - q_width
        : (kind == 2 ? output - q_width - gate_width : output - q_width - gate_width - SEISMIC_DIM_NV));
    float values[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    for (ulong source = ulong(lane); source < SEISMIC_DIM_H; source += ulong(simd_width)) {
        ulong logical = weight_row * SEISMIC_DIM_H + source;
        float weight = kind == 0 ? rp_qkv(qkv_weight, logical)
            : (kind == 1 ? rp_gate(gate_weight, logical)
            : (kind == 2 ? rp_alpha(alpha_weight, logical) : rp_beta(beta_weight, logical)));
        for (uint r = 0; r < 8; ++r) {
            ulong row = row_base + r;
            if (row < SEISMIC_DIM_M) {
                float x = rp_activation(normalized, rp2(row, source,
                    SEISMIC_NORMALIZED_STRIDE_0, SEISMIC_NORMALIZED_STRIDE_1));
                values[r] = metal::fma(x, weight, values[r]);
            }
        }
    }
    for (uint r = 0; r < 8; ++r) {
        ulong row = row_base + r;
        if (row >= SEISMIC_DIM_M) break;
        float projected = simd_sum(values[r]);
        if (lane == 0)
            rp_store_activation(projection, rp2(row, output, SEISMIC_RESULT_0_STRIDE_0,
                SEISMIC_RESULT_0_STRIDE_1), projected);
    }
}
