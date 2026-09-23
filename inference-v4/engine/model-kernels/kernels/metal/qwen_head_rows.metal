inline ulong head_input_at2(ulong row, ulong column, ulong stride0, ulong stride1) {
    return row * stride0 + column * stride1;
}

inline uint head_input_code(device const uchar *bytes, ulong bit, uint width) {
    uint value = 0;
    for (uint offset = 0; offset < width; ++offset)
        value |= uint((bytes[(bit + offset) >> 3] >> ((bit + offset) & 7)) & 1) << offset;
    return value;
}

inline float head_input_resident(device const uchar *base, ulong logical, uint kind,
    ulong packet_size, ulong group, ulong words, ulong coefficients, ulong factor, ulong bias) {
    device const uchar *packet = base + (logical / group) * packet_size;
    ulong position = logical % group;
    if (kind == 8)
        return float(int(reinterpret_cast<device const char *>(packet + words)[position]))
            * float(*reinterpret_cast<device const half *>(packet + factor));
    if (kind == 14) {
        const int table[16] = {-127,-104,-83,-65,-49,-35,-22,-10,1,13,25,38,53,69,89,113};
        return reinterpret_cast<device const float *>(packet + factor)[position / 32]
            * float(table[head_input_code(packet + words, position * 4, 4)]);
    }
    int code = int(head_input_code(packet + words, position * kind, kind));
    if (kind == 6) code -= 32;
    ulong ci = position / (kind == 6 ? 16 : 32);
    if (kind == 6)
        return float(code * int(reinterpret_cast<device const char *>(packet + coefficients)[ci]))
            * float(*reinterpret_cast<device const half *>(packet + factor));
    uint scale_code = head_input_code(packet + coefficients, ci * 12, 6);
    uint bias_code = head_input_code(packet + coefficients, ci * 12 + 6, 6);
    return metal::fma(float(*reinterpret_cast<device const half *>(packet + factor))
        * float(scale_code), float(code),
        -float(*reinterpret_cast<device const half *>(packet + bias)) * float(bias_code));
}

#define HEAD_DENSE_LOAD(PREFIX, BASE, LOGICAL) \
    *reinterpret_cast<device const PREFIX##_TYPE *>((BASE) + (LOGICAL) * PREFIX##_PACKET_SIZE)

inline float head_input_table(device const uchar *base, ulong logical) {
#if defined(SEISMIC_TABLE_REPRESENTATION_F32)
    return *reinterpret_cast<device const float *>(base + logical * SEISMIC_TABLE_PACKET_SIZE);
#elif defined(SEISMIC_TABLE_REPRESENTATION_F16)
    return float(*reinterpret_cast<device const half *>(base + logical * SEISMIC_TABLE_PACKET_SIZE));
#elif defined(SEISMIC_TABLE_REPRESENTATION_BF16)
    return as_type<float>(uint(*reinterpret_cast<device const ushort *>(base + logical * SEISMIC_TABLE_PACKET_SIZE)) << 16);
#elif defined(SEISMIC_TABLE_REPRESENTATION_Q8G32S)
    return head_input_resident(base, logical, 8, SEISMIC_TABLE_PACKET_SIZE, SEISMIC_TABLE_LOGICAL_GROUP, SEISMIC_TABLE_PLANE_0_OFFSET, 0, SEISMIC_TABLE_PLANE_1_OFFSET, 0);
#elif defined(SEISMIC_TABLE_REPRESENTATION_Q4K)
    return head_input_resident(base, logical, 4, SEISMIC_TABLE_PACKET_SIZE, SEISMIC_TABLE_LOGICAL_GROUP, SEISMIC_TABLE_PLANE_0_OFFSET, SEISMIC_TABLE_PLANE_1_OFFSET, SEISMIC_TABLE_PLANE_2_OFFSET, SEISMIC_TABLE_PLANE_3_OFFSET);
#elif defined(SEISMIC_TABLE_REPRESENTATION_Q5K)
    return head_input_resident(base, logical, 5, SEISMIC_TABLE_PACKET_SIZE, SEISMIC_TABLE_LOGICAL_GROUP, SEISMIC_TABLE_PLANE_0_OFFSET, SEISMIC_TABLE_PLANE_1_OFFSET, SEISMIC_TABLE_PLANE_2_OFFSET, SEISMIC_TABLE_PLANE_3_OFFSET);
#elif defined(SEISMIC_TABLE_REPRESENTATION_Q6K)
    return head_input_resident(base, logical, 6, SEISMIC_TABLE_PACKET_SIZE, SEISMIC_TABLE_LOGICAL_GROUP, SEISMIC_TABLE_PLANE_0_OFFSET, SEISMIC_TABLE_PLANE_1_OFFSET, SEISMIC_TABLE_PLANE_2_OFFSET, 0);
#elif defined(SEISMIC_TABLE_REPRESENTATION_IQ4G32)
    return head_input_resident(base, logical, 14, SEISMIC_TABLE_PACKET_SIZE, SEISMIC_TABLE_LOGICAL_GROUP, SEISMIC_TABLE_PLANE_0_OFFSET, 0, SEISMIC_TABLE_PLANE_1_OFFSET, 0);
#else
#error "unsupported draft-head embedding table"
#endif
}

#define HEAD_INPUT_DENSE_LOADER(NAME, PREFIX) \
inline float NAME(device const uchar *base, ulong logical) { \
    /* all norms and activation rows are dense by contract */ \
    if (PREFIX##_KIND == 0) return *reinterpret_cast<device const float *>(base + logical * PREFIX##_PACKET_SIZE); \
    if (PREFIX##_KIND == 1) return float(*reinterpret_cast<device const half *>(base + logical * PREFIX##_PACKET_SIZE)); \
    return as_type<float>(uint(*reinterpret_cast<device const ushort *>(base + logical * PREFIX##_PACKET_SIZE)) << 16); \
}

#if defined(SEISMIC_CONDITIONING_REPRESENTATION_F32)
#define CONDITIONING_KIND 0
#elif defined(SEISMIC_CONDITIONING_REPRESENTATION_F16)
#define CONDITIONING_KIND 1
#elif defined(SEISMIC_CONDITIONING_REPRESENTATION_BF16)
#define CONDITIONING_KIND 2
#else
#error "draft-head conditioning must be dense"
#endif
#define CONDITIONING_PACKET_SIZE SEISMIC_CONDITIONING_PACKET_SIZE
HEAD_INPUT_DENSE_LOADER(head_input_conditioning, CONDITIONING)

#if defined(SEISMIC_EMBEDDING_NORM_REPRESENTATION_F32)
#define EMBEDDING_NORM_KIND 0
#elif defined(SEISMIC_EMBEDDING_NORM_REPRESENTATION_F16)
#define EMBEDDING_NORM_KIND 1
#elif defined(SEISMIC_EMBEDDING_NORM_REPRESENTATION_BF16)
#define EMBEDDING_NORM_KIND 2
#else
#error "draft-head embedding norm must be dense"
#endif
#define EMBEDDING_NORM_PACKET_SIZE SEISMIC_EMBEDDING_NORM_PACKET_SIZE
HEAD_INPUT_DENSE_LOADER(head_input_embedding_norm, EMBEDDING_NORM)

#if defined(SEISMIC_HIDDEN_NORM_REPRESENTATION_F32)
#define HIDDEN_NORM_KIND 0
#elif defined(SEISMIC_HIDDEN_NORM_REPRESENTATION_F16)
#define HIDDEN_NORM_KIND 1
#elif defined(SEISMIC_HIDDEN_NORM_REPRESENTATION_BF16)
#define HIDDEN_NORM_KIND 2
#else
#error "draft-head hidden norm must be dense"
#endif
#define HIDDEN_NORM_PACKET_SIZE SEISMIC_HIDDEN_NORM_PACKET_SIZE
HEAD_INPUT_DENSE_LOADER(head_input_hidden_norm, HIDDEN_NORM)

inline float head_input_combine(device const uchar *base, ulong logical) {
#if defined(SEISMIC_COMBINE_REPRESENTATION_F32)
    return *reinterpret_cast<device const float *>(base + logical * SEISMIC_COMBINE_PACKET_SIZE);
#elif defined(SEISMIC_COMBINE_REPRESENTATION_F16)
    return float(*reinterpret_cast<device const half *>(base + logical * SEISMIC_COMBINE_PACKET_SIZE));
#elif defined(SEISMIC_COMBINE_REPRESENTATION_BF16)
    return as_type<float>(uint(*reinterpret_cast<device const ushort *>(base + logical * SEISMIC_COMBINE_PACKET_SIZE)) << 16);
#elif defined(SEISMIC_COMBINE_REPRESENTATION_Q8G32S)
    return head_input_resident(base, logical, 8, SEISMIC_COMBINE_PACKET_SIZE, SEISMIC_COMBINE_LOGICAL_GROUP, SEISMIC_COMBINE_PLANE_0_OFFSET, 0, SEISMIC_COMBINE_PLANE_1_OFFSET, 0);
#elif defined(SEISMIC_COMBINE_REPRESENTATION_Q4K)
    return head_input_resident(base, logical, 4, SEISMIC_COMBINE_PACKET_SIZE, SEISMIC_COMBINE_LOGICAL_GROUP, SEISMIC_COMBINE_PLANE_0_OFFSET, SEISMIC_COMBINE_PLANE_1_OFFSET, SEISMIC_COMBINE_PLANE_2_OFFSET, SEISMIC_COMBINE_PLANE_3_OFFSET);
#elif defined(SEISMIC_COMBINE_REPRESENTATION_Q5K)
    return head_input_resident(base, logical, 5, SEISMIC_COMBINE_PACKET_SIZE, SEISMIC_COMBINE_LOGICAL_GROUP, SEISMIC_COMBINE_PLANE_0_OFFSET, SEISMIC_COMBINE_PLANE_1_OFFSET, SEISMIC_COMBINE_PLANE_2_OFFSET, SEISMIC_COMBINE_PLANE_3_OFFSET);
#elif defined(SEISMIC_COMBINE_REPRESENTATION_Q6K)
    return head_input_resident(base, logical, 6, SEISMIC_COMBINE_PACKET_SIZE, SEISMIC_COMBINE_LOGICAL_GROUP, SEISMIC_COMBINE_PLANE_0_OFFSET, SEISMIC_COMBINE_PLANE_1_OFFSET, SEISMIC_COMBINE_PLANE_2_OFFSET, 0);
#elif defined(SEISMIC_COMBINE_REPRESENTATION_IQ4G32)
    return head_input_resident(base, logical, 14, SEISMIC_COMBINE_PACKET_SIZE, SEISMIC_COMBINE_LOGICAL_GROUP, SEISMIC_COMBINE_PLANE_0_OFFSET, 0, SEISMIC_COMBINE_PLANE_1_OFFSET, 0);
#else
#error "unsupported draft-head combine weight"
#endif
}

kernel void qwen_head_rows(
    device const int *tokens [[buffer(SEISMIC_BUFFER_TOKENS)]],
    device const uchar *table [[buffer(SEISMIC_BUFFER_TABLE)]],
    device const uchar *conditioning [[buffer(SEISMIC_BUFFER_CONDITIONING)]],
    device const uchar *embedding_norm [[buffer(SEISMIC_BUFFER_EMBEDDING_NORM)]],
    device const uchar *hidden_norm [[buffer(SEISMIC_BUFFER_HIDDEN_NORM)]],
    device const uchar *combine [[buffer(SEISMIC_BUFFER_COMBINE)]],
    device float *result [[buffer(SEISMIC_RESULT_0_BUFFER)]],
    constant ulong *seismic_words [[buffer(SEISMIC_BUFFER_WORDS)]],
    uint raw_index [[thread_position_in_grid]]) {
    ulong index = ulong(raw_index);
    if (index >= SEISMIC_DIM_M * SEISMIC_DIM_D) return;
    ulong row = index / SEISMIC_DIM_D;
    ulong output = index % SEISMIC_DIM_D;
    int token = tokens[row * SEISMIC_TOKENS_STRIDE_0];
    float embedding_squares = 0.0f;
    float hidden_squares = 0.0f;
    for (ulong column = 0; column < SEISMIC_DIM_D; ++column) {
        float embedded = head_input_table(table, ulong(token) * SEISMIC_DIM_D + column);
        float conditioned = head_input_conditioning(conditioning,
            head_input_at2(row, column, SEISMIC_CONDITIONING_STRIDE_0, SEISMIC_CONDITIONING_STRIDE_1));
        embedding_squares += embedded * embedded;
        hidden_squares += conditioned * conditioned;
    }
    float epsilon = as_type<float>(uint(SEISMIC_PARAM_EPSILON));
    float embedding_inverse = metal::rsqrt(embedding_squares / float(SEISMIC_DIM_D) + epsilon);
    float hidden_inverse = metal::rsqrt(hidden_squares / float(SEISMIC_DIM_D) + epsilon);
    float sum = 0.0f;
    for (ulong column = 0; column < SEISMIC_DIM_D; ++column) {
        float embedded = head_input_table(table, ulong(token) * SEISMIC_DIM_D + column)
            * embedding_inverse * head_input_embedding_norm(embedding_norm, column * SEISMIC_EMBEDDING_NORM_STRIDE_0);
        float conditioned = head_input_conditioning(conditioning,
            head_input_at2(row, column, SEISMIC_CONDITIONING_STRIDE_0, SEISMIC_CONDITIONING_STRIDE_1))
            * hidden_inverse * head_input_hidden_norm(hidden_norm, column * SEISMIC_HIDDEN_NORM_STRIDE_0);
        sum = metal::fma(embedded, head_input_combine(combine, output * (2 * SEISMIC_DIM_D) + column), sum);
        sum = metal::fma(conditioned, head_input_combine(combine, output * (2 * SEISMIC_DIM_D) + SEISMIC_DIM_D + column), sum);
    }
    result[head_input_at2(row, output, SEISMIC_RESULT_0_STRIDE_0, SEISMIC_RESULT_0_STRIDE_1)] = sum;
}
