inline float vision_dense(device const uchar *base, ulong logical, uint kind, ulong packet) {
    if (kind == 0) return *reinterpret_cast<device const float *>(base + logical * packet);
    if (kind == 1) return float(*reinterpret_cast<device const half *>(base + logical * packet));
    return as_type<float>(uint(*reinterpret_cast<device const ushort *>(base + logical * packet)) << 16);
}

#if defined(SEISMIC_ELEMENT_A_REPRESENTATION_F32)
#define VK_A 0
#elif defined(SEISMIC_ELEMENT_A_REPRESENTATION_F16)
#define VK_A 1
#elif defined(SEISMIC_ELEMENT_A_REPRESENTATION_BF16)
#define VK_A 2
#else
#error "vision block requires dense activations"
#endif
#if defined(SEISMIC_NORM1_WEIGHT_REPRESENTATION_F32)
#define VK_NW 0
#elif defined(SEISMIC_NORM1_WEIGHT_REPRESENTATION_F16)
#define VK_NW 1
#else
#define VK_NW 2
#endif
#if defined(SEISMIC_NORM1_BIAS_REPRESENTATION_F32)
#define VK_NB 0
#elif defined(SEISMIC_NORM1_BIAS_REPRESENTATION_F16)
#define VK_NB 1
#else
#define VK_NB 2
#endif
#if defined(SEISMIC_NORM2_WEIGHT_REPRESENTATION_F32)
#define VK_N2W 0
#elif defined(SEISMIC_NORM2_WEIGHT_REPRESENTATION_F16)
#define VK_N2W 1
#else
#define VK_N2W 2
#endif
#if defined(SEISMIC_NORM2_BIAS_REPRESENTATION_F32)
#define VK_N2B 0
#elif defined(SEISMIC_NORM2_BIAS_REPRESENTATION_F16)
#define VK_N2B 1
#else
#define VK_N2B 2
#endif
#if defined(SEISMIC_QKV_WEIGHT_REPRESENTATION_F32)
#define VK_QW 0
#elif defined(SEISMIC_QKV_WEIGHT_REPRESENTATION_F16)
#define VK_QW 1
#elif defined(SEISMIC_QKV_WEIGHT_REPRESENTATION_BF16)
#define VK_QW 2
#else
#error "native vision block currently requires dense resident QKV weights"
#endif
#if defined(SEISMIC_QKV_BIAS_REPRESENTATION_F32)
#define VK_QB 0
#elif defined(SEISMIC_QKV_BIAS_REPRESENTATION_F16)
#define VK_QB 1
#else
#define VK_QB 2
#endif
#if defined(SEISMIC_PROJECTION_WEIGHT_REPRESENTATION_F32)
#define VK_PW 0
#elif defined(SEISMIC_PROJECTION_WEIGHT_REPRESENTATION_F16)
#define VK_PW 1
#elif defined(SEISMIC_PROJECTION_WEIGHT_REPRESENTATION_BF16)
#define VK_PW 2
#else
#error "native vision block currently requires dense resident projection weights"
#endif
#if defined(SEISMIC_PROJECTION_BIAS_REPRESENTATION_F32)
#define VK_PB 0
#elif defined(SEISMIC_PROJECTION_BIAS_REPRESENTATION_F16)
#define VK_PB 1
#else
#define VK_PB 2
#endif
#if defined(SEISMIC_UP_WEIGHT_REPRESENTATION_F32)
#define VK_UW 0
#elif defined(SEISMIC_UP_WEIGHT_REPRESENTATION_F16)
#define VK_UW 1
#elif defined(SEISMIC_UP_WEIGHT_REPRESENTATION_BF16)
#define VK_UW 2
#else
#error "native vision block currently requires dense resident up weights"
#endif
#if defined(SEISMIC_UP_BIAS_REPRESENTATION_F32)
#define VK_UB 0
#elif defined(SEISMIC_UP_BIAS_REPRESENTATION_F16)
#define VK_UB 1
#else
#define VK_UB 2
#endif
#if defined(SEISMIC_DOWN_WEIGHT_REPRESENTATION_F32)
#define VK_DW 0
#elif defined(SEISMIC_DOWN_WEIGHT_REPRESENTATION_F16)
#define VK_DW 1
#elif defined(SEISMIC_DOWN_WEIGHT_REPRESENTATION_BF16)
#define VK_DW 2
#else
#error "native vision block currently requires dense resident down weights"
#endif
#if defined(SEISMIC_DOWN_BIAS_REPRESENTATION_F32)
#define VK_DB 0
#elif defined(SEISMIC_DOWN_BIAS_REPRESENTATION_F16)
#define VK_DB 1
#else
#define VK_DB 2
#endif

inline float hidden_at(device const uchar *hidden, ulong row, ulong column) {
    ulong head = column / (4 * SEISMIC_DIM_P);
    ulong rem = column % (4 * SEISMIC_DIM_P);
    ulong lane = rem / SEISMIC_DIM_P;
    ulong part = rem % SEISMIC_DIM_P;
    return vision_dense(hidden, row * SEISMIC_HIDDEN_STRIDE_0 + head * SEISMIC_HIDDEN_STRIDE_1
        + lane * SEISMIC_HIDDEN_STRIDE_2 + part * SEISMIC_HIDDEN_STRIDE_3,
        VK_A, SEISMIC_ELEMENT_A_PACKET_SIZE);
}

inline float normalized1(device const uchar *hidden, device const uchar *weight,
    device const uchar *bias, ulong row, ulong column, float epsilon) {
    const ulong width = SEISMIC_DIM_H * 4 * SEISMIC_DIM_P;
    float sum = 0.0f, squares = 0.0f;
    for (ulong i = 0; i < width; ++i) { float x = hidden_at(hidden, row, i); sum += x; squares += x * x; }
    float mean = sum / float(width);
    float variance = squares / float(width) - mean * mean;
    return (hidden_at(hidden, row, column) - mean) * metal::rsqrt(variance + epsilon)
        * vision_dense(weight, column * SEISMIC_NORM1_WEIGHT_STRIDE_0, VK_NW, SEISMIC_NORM1_WEIGHT_PACKET_SIZE)
        + vision_dense(bias, column * SEISMIC_NORM1_BIAS_STRIDE_0, VK_NB, SEISMIC_NORM1_BIAS_PACKET_SIZE);
}

inline float qkv_at(device const uchar *hidden, device const uchar *nw, device const uchar *nb,
    device const uchar *qw, device const uchar *qb, ulong row, ulong component, ulong column,
    float epsilon) {
    const ulong width = SEISMIC_DIM_H * 4 * SEISMIC_DIM_P;
    ulong output = component * width + column;
    float value = vision_dense(qb, output * SEISMIC_QKV_BIAS_STRIDE_0, VK_QB, SEISMIC_QKV_BIAS_PACKET_SIZE);
    for (ulong source = 0; source < width; ++source)
        value = metal::fma(normalized1(hidden, nw, nb, row, source, epsilon),
            vision_dense(qw, source * SEISMIC_QKV_WEIGHT_STRIDE_0 + output * SEISMIC_QKV_WEIGHT_STRIDE_1,
                VK_QW, SEISMIC_QKV_WEIGHT_PACKET_SIZE), value);
    return value;
}

inline float rotated_qk(device const uchar *hidden, device const int *coordinates,
    device const uchar *nw, device const uchar *nb, device const uchar *qw, device const uchar *qb,
    ulong row, ulong component, ulong head, ulong local, float epsilon) {
    ulong pair = local % (2 * SEISMIC_DIM_P);
    ulong axis = pair / SEISMIC_DIM_P;
    ulong frequency = pair % SEISMIC_DIM_P;
    float angle = float(coordinates[row * SEISMIC_COORDINATES_STRIDE_0 + axis * SEISMIC_COORDINATES_STRIDE_1])
        * metal::exp(-metal::log(10000.0f) * float(frequency) / float(SEISMIC_DIM_P));
    ulong base = head * 4 * SEISMIC_DIM_P;
    float direct = qkv_at(hidden, nw, nb, qw, qb, row, component, base + local, epsilon);
    if (local < 2 * SEISMIC_DIM_P) {
        float cross = qkv_at(hidden, nw, nb, qw, qb, row, component,
            base + local + 2 * SEISMIC_DIM_P, epsilon);
        return direct * metal::cos(angle) - cross * metal::sin(angle);
    }
    float cross = qkv_at(hidden, nw, nb, qw, qb, row, component,
        base + local - 2 * SEISMIC_DIM_P, epsilon);
    return direct * metal::cos(angle) + cross * metal::sin(angle);
}

inline float attended_at(device const uchar *hidden, device const int *coordinates,
    device const uchar *nw, device const uchar *nb, device const uchar *qw, device const uchar *qb,
    ulong row, ulong head, ulong local, float epsilon) {
    const ulong head_width = 4 * SEISMIC_DIM_P;
    float maximum = -INFINITY;
    for (ulong source_row = 0; source_row < SEISMIC_DIM_M; ++source_row) {
        float score = 0.0f;
        for (ulong i = 0; i < head_width; ++i)
            score += rotated_qk(hidden, coordinates, nw, nb, qw, qb, row, 0, head, i, epsilon)
                * rotated_qk(hidden, coordinates, nw, nb, qw, qb, source_row, 1, head, i, epsilon);
        maximum = metal::max(maximum, score * metal::rsqrt(float(head_width)));
    }
    float denominator = 0.0f, numerator = 0.0f;
    for (ulong source_row = 0; source_row < SEISMIC_DIM_M; ++source_row) {
        float score = 0.0f;
        for (ulong i = 0; i < head_width; ++i)
            score += rotated_qk(hidden, coordinates, nw, nb, qw, qb, row, 0, head, i, epsilon)
                * rotated_qk(hidden, coordinates, nw, nb, qw, qb, source_row, 1, head, i, epsilon);
        float probability = metal::exp(score * metal::rsqrt(float(head_width)) - maximum);
        denominator += probability;
        numerator += probability * qkv_at(hidden, nw, nb, qw, qb, source_row, 2,
            head * head_width + local, epsilon);
    }
    return numerator / denominator;
}

inline float residual_at(device const uchar *hidden, device const int *coordinates,
    device const uchar *nw, device const uchar *nb, device const uchar *qw, device const uchar *qb,
    device const uchar *pw, device const uchar *pb, ulong row, ulong column, float epsilon) {
    const ulong width = SEISMIC_DIM_H * 4 * SEISMIC_DIM_P;
    float mixed = vision_dense(pb, column * SEISMIC_PROJECTION_BIAS_STRIDE_0, VK_PB, SEISMIC_PROJECTION_BIAS_PACKET_SIZE);
    for (ulong source = 0; source < width; ++source) {
        ulong head = source / (4 * SEISMIC_DIM_P), local = source % (4 * SEISMIC_DIM_P);
        mixed = metal::fma(attended_at(hidden, coordinates, nw, nb, qw, qb, row, head, local, epsilon),
            vision_dense(pw, source * SEISMIC_PROJECTION_WEIGHT_STRIDE_0 + column * SEISMIC_PROJECTION_WEIGHT_STRIDE_1,
                VK_PW, SEISMIC_PROJECTION_WEIGHT_PACKET_SIZE), mixed);
    }
    return hidden_at(hidden, row, column) + mixed;
}

inline float normalized2(device const uchar *hidden, device const int *coordinates,
    device const uchar *n1w, device const uchar *n1b, device const uchar *qw, device const uchar *qb,
    device const uchar *pw, device const uchar *pb, device const uchar *n2w, device const uchar *n2b,
    ulong row, ulong column, float epsilon) {
    const ulong width = SEISMIC_DIM_H * 4 * SEISMIC_DIM_P;
    float sum = 0.0f, squares = 0.0f;
    for (ulong i = 0; i < width; ++i) {
        float x = residual_at(hidden, coordinates, n1w, n1b, qw, qb, pw, pb, row, i, epsilon);
        sum += x; squares += x * x;
    }
    float mean = sum / float(width), variance = squares / float(width) - mean * mean;
    return (residual_at(hidden, coordinates, n1w, n1b, qw, qb, pw, pb, row, column, epsilon) - mean)
        * metal::rsqrt(variance + epsilon)
        * vision_dense(n2w, column * SEISMIC_NORM2_WEIGHT_STRIDE_0, VK_N2W, SEISMIC_NORM2_WEIGHT_PACKET_SIZE)
        + vision_dense(n2b, column * SEISMIC_NORM2_BIAS_STRIDE_0, VK_N2B, SEISMIC_NORM2_BIAS_PACKET_SIZE);
}

kernel void qwen_vision_block(
    device const uchar *hidden [[buffer(SEISMIC_BUFFER_HIDDEN)]],
    device const int *coordinates [[buffer(SEISMIC_BUFFER_COORDINATES)]],
    device const uchar *norm1_weight [[buffer(SEISMIC_BUFFER_NORM1_WEIGHT)]],
    device const uchar *norm1_bias [[buffer(SEISMIC_BUFFER_NORM1_BIAS)]],
    device const uchar *qkv_weight [[buffer(SEISMIC_BUFFER_QKV_WEIGHT)]],
    device const uchar *qkv_bias [[buffer(SEISMIC_BUFFER_QKV_BIAS)]],
    device const uchar *projection_weight [[buffer(SEISMIC_BUFFER_PROJECTION_WEIGHT)]],
    device const uchar *projection_bias [[buffer(SEISMIC_BUFFER_PROJECTION_BIAS)]],
    device const uchar *norm2_weight [[buffer(SEISMIC_BUFFER_NORM2_WEIGHT)]],
    device const uchar *norm2_bias [[buffer(SEISMIC_BUFFER_NORM2_BIAS)]],
    device const uchar *up_weight [[buffer(SEISMIC_BUFFER_UP_WEIGHT)]],
    device const uchar *up_bias [[buffer(SEISMIC_BUFFER_UP_BIAS)]],
    device const uchar *down_weight [[buffer(SEISMIC_BUFFER_DOWN_WEIGHT)]],
    device const uchar *down_bias [[buffer(SEISMIC_BUFFER_DOWN_BIAS)]],
    device uchar *result [[buffer(SEISMIC_RESULT_0_BUFFER)]],
    constant ulong *seismic_words [[buffer(SEISMIC_BUFFER_WORDS)]],
    uint raw_index [[thread_position_in_grid]]) {
    const ulong width = SEISMIC_DIM_H * 4 * SEISMIC_DIM_P;
    ulong index = ulong(raw_index);
    if (index >= SEISMIC_DIM_M * width) return;
    ulong row = index / width, column = index % width;
    float epsilon = as_type<float>(uint(SEISMIC_PARAM_EPSILON));
    float value = residual_at(hidden, coordinates, norm1_weight, norm1_bias, qkv_weight, qkv_bias,
        projection_weight, projection_bias, row, column, epsilon)
        + vision_dense(down_bias, column * SEISMIC_DOWN_BIAS_STRIDE_0, VK_DB, SEISMIC_DOWN_BIAS_PACKET_SIZE);
    for (ulong feature = 0; feature < SEISMIC_DIM_F; ++feature) {
        float up = vision_dense(up_bias, feature * SEISMIC_UP_BIAS_STRIDE_0, VK_UB, SEISMIC_UP_BIAS_PACKET_SIZE);
        for (ulong source = 0; source < width; ++source)
            up = metal::fma(normalized2(hidden, coordinates, norm1_weight, norm1_bias, qkv_weight,
                qkv_bias, projection_weight, projection_bias, norm2_weight, norm2_bias,
                row, source, epsilon), vision_dense(up_weight,
                    source * SEISMIC_UP_WEIGHT_STRIDE_0 + feature * SEISMIC_UP_WEIGHT_STRIDE_1,
                    VK_UW, SEISMIC_UP_WEIGHT_PACKET_SIZE), up);
        float activated = 0.5f * up * (1.0f + metal::tanh(0.7978845608028654f
            * (up + 0.044715f * up * up * up)));
        value = metal::fma(activated, vision_dense(down_weight,
            feature * SEISMIC_DOWN_WEIGHT_STRIDE_0 + column * SEISMIC_DOWN_WEIGHT_STRIDE_1,
            VK_DW, SEISMIC_DOWN_WEIGHT_PACKET_SIZE), value);
    }
#if defined(SEISMIC_ELEMENT_A_REPRESENTATION_F32)
    *reinterpret_cast<device float *>(result + (row * SEISMIC_RESULT_0_STRIDE_0 + column * SEISMIC_RESULT_0_STRIDE_1) * SEISMIC_ELEMENT_A_PACKET_SIZE) = value;
#elif defined(SEISMIC_ELEMENT_A_REPRESENTATION_F16)
    *reinterpret_cast<device half *>(result + (row * SEISMIC_RESULT_0_STRIDE_0 + column * SEISMIC_RESULT_0_STRIDE_1) * SEISMIC_ELEMENT_A_PACKET_SIZE) = half(value);
#else
    uint bits = as_type<uint>(value);
    *reinterpret_cast<device ushort *>(result + (row * SEISMIC_RESULT_0_STRIDE_0 + column * SEISMIC_RESULT_0_STRIDE_1) * SEISMIC_ELEMENT_A_PACKET_SIZE) = ushort((bits + 0x7fffu + ((bits >> 16) & 1u)) >> 16);
#endif
}
