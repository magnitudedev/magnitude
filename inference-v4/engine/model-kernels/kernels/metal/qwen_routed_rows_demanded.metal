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
inline float load_norm(device const uchar *base, ulong logical) {
#if defined(SEISMIC_NORM_REPRESENTATION_F32)
    return XDENSE(SEISMIC_NORM, 0);
#elif defined(SEISMIC_NORM_REPRESENTATION_F16)
    return XDENSE(SEISMIC_NORM, 1);
#elif defined(SEISMIC_NORM_REPRESENTATION_BF16)
    return XDENSE(SEISMIC_NORM, 2);
#else
#error "unsupported routed norm representation"
#endif
}
inline float load_router(device const uchar *base, ulong logical) {
#if defined(SEISMIC_ROUTER_REPRESENTATION_F32)
    return XDENSE(SEISMIC_ROUTER, 0);
#elif defined(SEISMIC_ROUTER_REPRESENTATION_F16)
    return XDENSE(SEISMIC_ROUTER, 1);
#elif defined(SEISMIC_ROUTER_REPRESENTATION_BF16)
    return XDENSE(SEISMIC_ROUTER, 2);
#elif defined(SEISMIC_ROUTER_REPRESENTATION_Q8G32S)
    return XQ8(SEISMIC_ROUTER);
#elif defined(SEISMIC_ROUTER_REPRESENTATION_Q4K)
    return XQK(SEISMIC_ROUTER, 4);
#elif defined(SEISMIC_ROUTER_REPRESENTATION_Q5K)
    return XQK(SEISMIC_ROUTER, 5);
#elif defined(SEISMIC_ROUTER_REPRESENTATION_Q6K)
    return XQ6(SEISMIC_ROUTER);
#elif defined(SEISMIC_ROUTER_REPRESENTATION_IQ4G32)
    return XIQ(SEISMIC_ROUTER);
#else
#error "unsupported routed router representation"
#endif
}
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
inline float normalized_at(constant ulong *seismic_words, device const float *residual, device const uchar *norm,
    ulong row, ulong source, float inverse) {
    return round_activation(residual[offset2(row, source, SEISMIC_SOURCE_STRIDE_0, SEISMIC_SOURCE_STRIDE_1)]
        * inverse * load_norm(norm, source * SEISMIC_NORM_STRIDE_0));
}
inline float router_logit(constant ulong *seismic_words, device const float *residual, device const uchar *norm,
    device const uchar *router, ulong row, ulong expert, float inverse) {
    float sum = 0.0f;
    for (ulong source = 0; source < SEISMIC_DIM_H; ++source) {
        sum = metal::fma(normalized_at(seismic_words, residual, norm, row, source, inverse),
            load_router(router, expert * SEISMIC_DIM_H + source), sum);
    }
    return sum;
}
kernel void qwen_routed_rows_demanded(
    device const float *residual [[buffer(SEISMIC_BUFFER_SOURCE)]],
    device const int *out_rows [[buffer(SEISMIC_BUFFER_OUT_ROWS)]],
    device const uchar *norm [[buffer(SEISMIC_BUFFER_NORM)]],
    device const uchar *router [[buffer(SEISMIC_BUFFER_ROUTER)]],
    device const float *shared_router [[buffer(SEISMIC_BUFFER_SHARED_ROUTER)]],
    device const uchar *expert_gate [[buffer(SEISMIC_BUFFER_EXPERT_GATE)]],
    device const uchar *expert_up [[buffer(SEISMIC_BUFFER_EXPERT_UP)]],
    device const uchar *expert_down [[buffer(SEISMIC_BUFFER_EXPERT_DOWN)]],
    device const uchar *shared_gate [[buffer(SEISMIC_BUFFER_SHARED_GATE)]],
    device const uchar *shared_up [[buffer(SEISMIC_BUFFER_SHARED_UP)]],
    device const uchar *shared_down [[buffer(SEISMIC_BUFFER_SHARED_DOWN)]],
    device int *routes [[buffer(SEISMIC_BUFFER_ROUTES)]],
    device float *scores [[buffer(SEISMIC_BUFFER_SCORES)]],
    device float *result [[buffer(SEISMIC_RESULT_0_BUFFER)]],
    constant ulong *seismic_words [[buffer(SEISMIC_BUFFER_WORDS)]],
    uint index [[thread_position_in_grid]]) {
    if (ulong(index) >= SEISMIC_DIM_O * SEISMIC_DIM_H) return;
    ulong row = ulong(index) / SEISMIC_DIM_H;
    ulong source_row = ulong(out_rows[row * SEISMIC_OUT_ROWS_STRIDE_0]);
    ulong column = ulong(index) % SEISMIC_DIM_H;
    float squares = 0.0f;
    for (ulong source = 0; source < SEISMIC_DIM_H; ++source) {
        float value = residual[offset2(source_row, source, SEISMIC_SOURCE_STRIDE_0,
            SEISMIC_SOURCE_STRIDE_1)];
        squares += value * value;
    }
    float inverse = metal::rsqrt(squares / float(SEISMIC_DIM_H)
        + as_type<float>(uint(SEISMIC_PARAM_EPS)));
    float maximum = -INFINITY;
    for (ulong expert = 0; expert < SEISMIC_DIM_E; ++expert)
        maximum = metal::max(maximum, router_logit(seismic_words, residual, norm, router, source_row, expert, inverse));
    float total = 0.0f;
    for (ulong expert = 0; expert < SEISMIC_DIM_E; ++expert)
        total += metal::exp(router_logit(seismic_words, residual, norm, router, source_row, expert, inverse) - maximum);
    // Walk the probability/expert ordering in both directions. The scalar
    // boundary removes a fixed K-sized thread array from this oracle entry.
    float selected_total = 0.0f;
    float boundary_probability = INFINITY;
    int boundary_expert = 2147483647;
    for (ulong rank = 0; rank < SEISMIC_DIM_K; ++rank) {
        float best = -INFINITY;
        int winner = -1;
        for (ulong expert = 0; expert < SEISMIC_DIM_E; ++expert) {
            float probability = metal::exp(router_logit(seismic_words, residual, norm, router, source_row, expert, inverse) - maximum) / total;
            bool below_boundary = probability < boundary_probability
                || (probability == boundary_probability && int(expert) < boundary_expert);
            if (below_boundary && (probability > best
                    || (probability == best && int(expert) > winner))) {
                best = probability;
                winner = int(expert);
            }
        }
        selected_total += best;
        boundary_probability = best;
        boundary_expert = winner;
    }
    float expert_sum = 0.0f;
    for (ulong choice = 0; choice < SEISMIC_DIM_K; ++choice) {
        ulong expert = ulong(boundary_expert);
        float score = boundary_probability;
        if (SEISMIC_PARAM_NORMALIZE != 0) score /= selected_total;
        if (column == 0) {
            routes[offset2(row, choice, SEISMIC_ROUTES_STRIDE_0, SEISMIC_ROUTES_STRIDE_1)] = int(expert);
            scores[offset2(row, choice, SEISMIC_SCORES_STRIDE_0, SEISMIC_SCORES_STRIDE_1)] = score;
        }
        float projected = 0.0f;
        for (ulong feature = 0; feature < SEISMIC_DIM_F; ++feature) {
            float gate = 0.0f;
            float up = 0.0f;
            for (ulong source = 0; source < SEISMIC_DIM_H; ++source) {
                float input = normalized_at(seismic_words, residual, norm, source_row, source, inverse);
                ulong logical = (expert * SEISMIC_DIM_F + feature) * SEISMIC_DIM_H + source;
                gate = metal::fma(input, load_expert_gate(expert_gate, logical), gate);
                up = metal::fma(input, load_expert_up(expert_up, logical), up);
            }
            float product = round_activation(round_activation(gate / (1.0f + metal::exp(-gate))) * round_activation(up));
            projected = metal::fma(product, load_expert_down(expert_down,
                (expert * SEISMIC_DIM_H + column) * SEISMIC_DIM_F + feature), projected);
        }
        expert_sum = metal::fma(score, projected, expert_sum);
        // The next output choice is the immediate predecessor in descending
        // probability order, including the higher expert-id tie break.
        if (choice + 1 < SEISMIC_DIM_K) {
            float next_probability = INFINITY;
            int next_expert = 2147483647;
            for (ulong candidate = 0; candidate < SEISMIC_DIM_E; ++candidate) {
                float probability = metal::exp(router_logit(seismic_words, residual, norm, router, source_row, candidate, inverse) - maximum) / total;
                bool above_boundary = probability > boundary_probability
                    || (probability == boundary_probability && int(candidate) > boundary_expert);
                if (above_boundary && (probability < next_probability
                        || (probability == next_probability && int(candidate) < next_expert))) {
                    next_probability = probability;
                    next_expert = int(candidate);
                }
            }
            boundary_probability = next_probability;
            boundary_expert = next_expert;
        }
    }
    float shared_dot = 0.0f;
    for (ulong source = 0; source < SEISMIC_DIM_H; ++source)
        shared_dot = metal::fma(normalized_at(seismic_words, residual, norm, source_row, source, inverse),
            shared_router[source * SEISMIC_SHARED_ROUTER_STRIDE_0], shared_dot);
    float shared = 0.0f;
    for (ulong feature = 0; feature < SEISMIC_DIM_S; ++feature) {
        float gate = 0.0f;
        float up = 0.0f;
        for (ulong source = 0; source < SEISMIC_DIM_H; ++source) {
            float input = normalized_at(seismic_words, residual, norm, source_row, source, inverse);
            gate = metal::fma(input, load_shared_gate(shared_gate, feature * SEISMIC_DIM_H + source), gate);
            up = metal::fma(input, load_shared_up(shared_up, feature * SEISMIC_DIM_H + source), up);
        }
        float product = round_activation(round_activation(gate / (1.0f + metal::exp(-gate))) * round_activation(up));
        shared = metal::fma(product, load_shared_down(shared_down,
            column * SEISMIC_DIM_S + feature), shared);
    }
    result[offset2(row, column, SEISMIC_RESULT_0_STRIDE_0, SEISMIC_RESULT_0_STRIDE_1)] =
        residual[offset2(source_row, column, SEISMIC_SOURCE_STRIDE_0, SEISMIC_SOURCE_STRIDE_1)]
        + expert_sum + shared * (1.0f / (1.0f + metal::exp(-shared_dot)));
}
