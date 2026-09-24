// qwen_vision_block: one Qwen3-VL vision transformer block over M patch rows
// of D = H * W (W = 4P), the residual stream in F32. Eight launches (bodies in
// common/vision.h and the projection library): layer norm, QKV GEMM (+ bias),
// 2D rotation of the queries and keys in place, full attention, output GEMM
// (+ bias, + the block input), layer norm, up GEMM (+ bias, tanh GELU), down
// GEMM (+ bias, + the attention residual).
#include "common/vision.h"

#if defined(SEISMIC_ELEMENT_A_REPRESENTATION_BF16)
typedef element::Bf16 activation;
#elif defined(SEISMIC_ELEMENT_A_REPRESENTATION_F16)
typedef element::F16 activation;
#else
#error "qwen_vision_block requires a bf16 or f16 activation"
#endif

typedef vision::dense_packet<ELEMENT_KIND(SEISMIC_QKV_WEIGHT)>::type qkv_packet;
typedef vision::dense_packet<ELEMENT_KIND(SEISMIC_PROJECTION_WEIGHT)>::type projection_packet;
typedef vision::dense_packet<ELEMENT_KIND(SEISMIC_UP_WEIGHT)>::type up_packet;
typedef vision::dense_packet<ELEMENT_KIND(SEISMIC_DOWN_WEIGHT)>::type down_packet;
typedef ELEMENT_OF(SEISMIC_QKV_WEIGHT) qkv_element;
typedef ELEMENT_OF(SEISMIC_PROJECTION_WEIGHT) projection_element;
typedef ELEMENT_OF(SEISMIC_UP_WEIGHT) up_element;
typedef ELEMENT_OF(SEISMIC_DOWN_WEIGHT) down_element;
typedef ELEMENT_OF(SEISMIC_QKV_BIAS) qkv_bias_element;
typedef ELEMENT_OF(SEISMIC_PROJECTION_BIAS) projection_bias_element;
typedef ELEMENT_OF(SEISMIC_UP_BIAS) up_bias_element;
typedef ELEMENT_OF(SEISMIC_DOWN_BIAS) down_bias_element;
typedef ELEMENT_OF(SEISMIC_NORM1_WEIGHT) norm1_weight_element;
typedef ELEMENT_OF(SEISMIC_NORM1_BIAS) norm1_bias_element;
typedef ELEMENT_OF(SEISMIC_NORM2_WEIGHT) norm2_weight_element;
typedef ELEMENT_OF(SEISMIC_NORM2_BIAS) norm2_bias_element;

#define VISION_W (4 * SEISMIC_DIM_P)
#define VISION_D (SEISMIC_DIM_H * VISION_W)

#define BLOCK_ARGUMENTS                                                                          \
    device const float *hidden [[buffer(SEISMIC_BUFFER_HIDDEN)]],                                \
    device const int *coordinates [[buffer(SEISMIC_BUFFER_COORDINATES)]],                        \
    device const uchar *norm1_weight [[buffer(SEISMIC_BUFFER_NORM1_WEIGHT)]],                    \
    device const uchar *norm1_bias [[buffer(SEISMIC_BUFFER_NORM1_BIAS)]],                        \
    device const uchar *qkv_weight [[buffer(SEISMIC_BUFFER_QKV_WEIGHT)]],                        \
    device const uchar *qkv_bias [[buffer(SEISMIC_BUFFER_QKV_BIAS)]],                            \
    device const uchar *projection_weight [[buffer(SEISMIC_BUFFER_PROJECTION_WEIGHT)]],          \
    device const uchar *projection_bias [[buffer(SEISMIC_BUFFER_PROJECTION_BIAS)]],              \
    device const uchar *norm2_weight [[buffer(SEISMIC_BUFFER_NORM2_WEIGHT)]],                    \
    device const uchar *norm2_bias [[buffer(SEISMIC_BUFFER_NORM2_BIAS)]],                        \
    device const uchar *up_weight [[buffer(SEISMIC_BUFFER_UP_WEIGHT)]],                          \
    device const uchar *up_bias [[buffer(SEISMIC_BUFFER_UP_BIAS)]],                              \
    device const uchar *down_weight [[buffer(SEISMIC_BUFFER_DOWN_WEIGHT)]],                      \
    device const uchar *down_bias [[buffer(SEISMIC_BUFFER_DOWN_BIAS)]],                          \
    device float *result [[buffer(SEISMIC_RESULT_0_BUFFER)]],                                    \
    device uchar *normalized [[buffer(SEISMIC_BUFFER_SCRATCH_NORMALIZED)]],                      \
    device uchar *projected [[buffer(SEISMIC_BUFFER_SCRATCH_PROJECTED)]],                        \
    device uchar *attended [[buffer(SEISMIC_BUFFER_SCRATCH_ATTENDED)]],                          \
    device float *residual [[buffer(SEISMIC_BUFFER_SCRATCH_RESIDUAL)]],                          \
    device uchar *activated [[buffer(SEISMIC_BUFFER_SCRATCH_ACTIVATED)]],                        \
    constant ulong *seismic_words [[buffer(SEISMIC_BUFFER_WORDS)]]

#define NORM_THREADS 256

kernel void qwen_vision_block_norm1(BLOCK_ARGUMENTS,
    uint row [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float partials[NORM_THREADS / 32];
    vision::layer_norm<NORM_THREADS, activation, norm1_weight_element, norm1_bias_element>(
        hidden + ulong(row) * VISION_D, normalized + ulong(row) * VISION_D * activation::bytes, norm1_weight,
        norm1_bias, VISION_D, as_type<float>(uint(SEISMIC_PARAM_EPSILON)), partials, thread_index, sg, lane);
}

kernel void qwen_vision_block_qkv(BLOCK_ARGUMENTS,
    uint3 tile [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    PROJECTION_GEMM_SHARED(shared, vision::TILE_M, vision::TILE_N);
    projection::Plain<activation, projection::AllRows> in{normalized, VISION_D, 1, VISION_D, {}};
    vision::output_bias<activation, qkv_bias_element> out{projected, 3 * VISION_D, qkv_bias};
    auto w = vision::weight<qkv_packet>(qkv_weight, SEISMIC_QKV_WEIGHT_STRIDE_0, qkv_element::bytes, VISION_D);
    projection::gemm<qkv_packet, vision::TILE_M, vision::TILE_N>(in, out, w, uint(SEISMIC_DIM_M),
        3 * VISION_D, VISION_D, tile.y, tile.x, shared, sg, lane);
}

// One simdgroup per (row, query or key, head).
kernel void qwen_vision_block_rotate(BLOCK_ARGUMENTS,
    uint group [[threadgroup_position_in_grid]],
    uint simd [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    const ulong item = ulong(group) * 8 + simd;
    const ulong row = item / (2 * SEISMIC_DIM_H);
    if (row >= SEISMIC_DIM_M)
        return;
    const ulong part = item % (2 * SEISMIC_DIM_H);   // query heads, then key heads
    device typename activation::storage *head_row =
        reinterpret_cast<device typename activation::storage *>(projected) + row * 3 * VISION_D
        + part * VISION_W;
    vision::rotate<activation, VISION_W>(head_row, coordinates + row * 2, lane);
}

kernel void qwen_vision_block_attend(BLOCK_ARGUMENTS,
    uint3 group [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint simd [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    typedef typename activation::native S;
    threadgroup S keys[vision::ATTEND_KEYS * vision::attend_pitch<VISION_W>()];
    threadgroup S values[vision::ATTEND_KEYS * vision::attend_pitch<VISION_W>()];
    vision::attend<S, VISION_W>(reinterpret_cast<device const S *>(projected), reinterpret_cast<device S *>(attended),
        uint(SEISMIC_DIM_M), SEISMIC_DIM_H, metal::rsqrt(float(VISION_W)), keys, values, group.x, group.y,
        thread_index, simd, lane);
}

kernel void qwen_vision_block_output(BLOCK_ARGUMENTS,
    uint3 tile [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    PROJECTION_GEMM_SHARED(shared, vision::TILE_M, vision::TILE_N);
    projection::Plain<activation, projection::AllRows> in{attended, VISION_D, 1, VISION_D, {}};
    vision::output_bias_f32<projection_bias_element> out{residual, VISION_D, projection_bias, hidden, VISION_D};
    auto w = vision::weight<projection_packet>(projection_weight, SEISMIC_PROJECTION_WEIGHT_STRIDE_0,
        projection_element::bytes, VISION_D);
    projection::gemm<projection_packet, vision::TILE_M, vision::TILE_N>(in, out, w, uint(SEISMIC_DIM_M),
        VISION_D, VISION_D, tile.y, tile.x, shared, sg, lane);
}

kernel void qwen_vision_block_norm2(BLOCK_ARGUMENTS,
    uint row [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float partials[NORM_THREADS / 32];
    vision::layer_norm<NORM_THREADS, activation, norm2_weight_element, norm2_bias_element>(
        residual + ulong(row) * VISION_D, normalized + ulong(row) * VISION_D * activation::bytes, norm2_weight,
        norm2_bias, VISION_D, as_type<float>(uint(SEISMIC_PARAM_EPSILON)), partials, thread_index, sg, lane);
}

kernel void qwen_vision_block_up(BLOCK_ARGUMENTS,
    uint3 tile [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    PROJECTION_GEMM_SHARED(shared, vision::TILE_M, vision::TILE_N);
    const uint f = uint(SEISMIC_DIM_F);
    projection::Plain<activation, projection::AllRows> in{normalized, VISION_D, 1, VISION_D, {}};
    vision::output_bias_gelu<activation, up_bias_element, false> out{activated, f, up_bias};
    auto w = vision::weight<up_packet>(up_weight, SEISMIC_UP_WEIGHT_STRIDE_0, up_element::bytes, VISION_D);
    projection::gemm<up_packet, vision::TILE_M, vision::TILE_N>(in, out, w, uint(SEISMIC_DIM_M), f, VISION_D,
        tile.y, tile.x, shared, sg, lane);
}

kernel void qwen_vision_block_down(BLOCK_ARGUMENTS,
    uint3 tile [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    PROJECTION_GEMM_SHARED(shared, vision::TILE_M, vision::TILE_N);
    const uint f = uint(SEISMIC_DIM_F);
    projection::Plain<activation, projection::AllRows> in{activated, f, 1, f, {}};
    vision::output_bias_f32<down_bias_element> out{result, SEISMIC_RESULT_0_STRIDE_0, down_bias, residual,
        VISION_D};
    auto w = vision::weight<down_packet>(down_weight, SEISMIC_DOWN_WEIGHT_STRIDE_0, down_element::bytes, f);
    projection::gemm<down_packet, vision::TILE_M, vision::TILE_N>(in, out, w, uint(SEISMIC_DIM_M), VISION_D, f,
        tile.y, tile.x, shared, sg, lane);
}
