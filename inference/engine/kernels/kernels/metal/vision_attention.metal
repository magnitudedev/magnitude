// vision_attention: the prepare launch writes each row's queries and keys
// (head-normalized with NQ, rotated) and its values (normalized with NV) into
// the [M, H + 2KV, WP] operand rows: H query heads, then KV key heads, then KV
// value heads (W = 4P, WP = W rounded up to 16, the pad zero); the attend
// launch runs 64 query rows of one head per threadgroup over the keys of their
// spans. Bodies in lib/vision/vision.h.
#include "lib/vision/vision.h"

#if defined(SEISMIC_ELEMENT_A_REPRESENTATION_BF16)
typedef element::Bf16 activation;
#elif defined(SEISMIC_ELEMENT_A_REPRESENTATION_F16)
typedef element::F16 activation;
#else
#error "vision_attention requires a bf16 or f16 activation"
#endif

#define VISION_H (SEISMIC_DIM_KV * SEISMIC_DIM_G)
#define VISION_SLOTS (VISION_H + 2 * SEISMIC_DIM_KV)
#define VISION_W (4 * SEISMIC_DIM_P)
#define VISION_WP ((VISION_W + 15) / 16 * 16)

#define ATTENTION_ARGUMENTS                                                                      \
    device const uchar *query [[buffer(SEISMIC_BUFFER_QUERY)]],                                  \
    device const uchar *key [[buffer(SEISMIC_BUFFER_KEY)]],                                      \
    device const uchar *value [[buffer(SEISMIC_BUFFER_VALUE)]],                                  \
    device const float *query_norm [[buffer(SEISMIC_BUFFER_QUERY_NORM)]],                        \
    device const float *key_norm [[buffer(SEISMIC_BUFFER_KEY_NORM)]],                            \
    device const int *coordinates [[buffer(SEISMIC_BUFFER_COORDINATES)]],                        \
    device const int *spans [[buffer(SEISMIC_BUFFER_SPANS)]],                                    \
    device uchar *result [[buffer(SEISMIC_RESULT_0_BUFFER)]],                                    \
    device uchar *operands [[buffer(SEISMIC_BUFFER_SCRATCH_OPERANDS)]],                          \
    constant ulong *seismic_words [[buffer(SEISMIC_BUFFER_WORDS)]]

// One simdgroup per (row, head row): queries and keys (normalized with NQ)
// rotated, values normalized with NV (weightless) or copied.
kernel void vision_attention_prepare(ATTENTION_ARGUMENTS,
    uint group [[threadgroup_position_in_grid]],
    uint simd [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    typedef typename activation::storage S;
    const ulong item = ulong(group) * 8 + simd;
    const ulong row = item / VISION_SLOTS;
    if (row >= SEISMIC_DIM_M)
        return;
    const ulong slot = item % VISION_SLOTS;
    const ulong part = slot < VISION_H ? 0 : slot < VISION_H + SEISMIC_DIM_KV ? 1 : 2;
    device const uchar *source = part == 0 ? query : part == 1 ? key : value;
    const ulong heads = part == 0 ? VISION_H : SEISMIC_DIM_KV;
    const ulong head = part == 0 ? slot : slot - VISION_H - (part - 1) * SEISMIC_DIM_KV;
    device const S *from = reinterpret_cast<device const S *>(source) + (row * heads + head) * VISION_W;
    device S *to = reinterpret_cast<device S *>(operands) + (row * VISION_SLOTS + slot) * VISION_WP;
    const bool normed = part < 2 ? SEISMIC_DIM_NQ == 1 : SEISMIC_DIM_NV == 1;
    device const float *norm = part == 0 ? query_norm : part == 1 ? key_norm : nullptr;
    vision::prepare_head<activation, VISION_W, VISION_WP>(from, to, normed, norm,
        as_type<float>(uint(SEISMIC_PARAM_EPSILON)), part < 2 && as_type<float>(uint(SEISMIC_PARAM_LOG_BASE)) != 0.0f, coordinates + row * 2,
        as_type<float>(uint(SEISMIC_PARAM_LOG_BASE)), lane);
}

kernel void vision_attention_attend(ATTENTION_ARGUMENTS,
    uint3 group [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint simd [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    typedef typename activation::native S;
    threadgroup S keys[vision::ATTEND_KEYS * vision::attend_pitch<VISION_WP>()];
    threadgroup S values[vision::ATTEND_KEYS * vision::attend_pitch<VISION_WP>()];
    threadgroup uint bounds[2];
    const float scale = int(SEISMIC_PARAM_UNIT_SCALE) != 0 ? 1.0f : metal::rsqrt(float(VISION_W));
#if SEISMIC_HAS_TENSOR_OPS
    ATTEND_TENSOR_SLOTS(slots);
    vision::attend_tensor<S, VISION_W, VISION_WP>(reinterpret_cast<device const S *>(operands),
        reinterpret_cast<device S *>(result), uint(SEISMIC_DIM_M), VISION_H, SEISMIC_DIM_G, scale,
        SEISMIC_DIM_WS == 1 ? spans : nullptr, keys, values, bounds, slots, group.x, group.y, thread_index, simd,
        lane);
#else
    vision::attend<S, VISION_W, VISION_WP>(reinterpret_cast<device const S *>(operands),
        reinterpret_cast<device S *>(result), uint(SEISMIC_DIM_M), VISION_H, SEISMIC_DIM_G, scale,
        SEISMIC_DIM_WS == 1 ? spans : nullptr, keys, values, bounds, group.x, group.y, thread_index, simd, lane);
#endif
}
