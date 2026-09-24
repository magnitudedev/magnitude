// qwen_vision_merger: layer norm of each of the M * G F32 patch rows (width
// H) to A, then the up GEMM over G concatenated rows (+ bias, erf GELU) and
// the down GEMM to the decoder width D (+ bias), published in F32 as the
// image features. Bodies in common/vision.h and the projection library.
#include "common/vision.h"

#if defined(SEISMIC_ELEMENT_A_REPRESENTATION_BF16)
typedef element::Bf16 activation;
#elif defined(SEISMIC_ELEMENT_A_REPRESENTATION_F16)
typedef element::F16 activation;
#else
#error "qwen_vision_merger requires a bf16 or f16 activation"
#endif

typedef vision::dense_packet<ELEMENT_KIND(SEISMIC_UP_WEIGHT)>::type up_packet;
typedef vision::dense_packet<ELEMENT_KIND(SEISMIC_DOWN_WEIGHT)>::type down_packet;
typedef ELEMENT_OF(SEISMIC_UP_WEIGHT) up_element;
typedef ELEMENT_OF(SEISMIC_DOWN_WEIGHT) down_element;
typedef ELEMENT_OF(SEISMIC_UP_BIAS) up_bias_element;
typedef ELEMENT_OF(SEISMIC_DOWN_BIAS) down_bias_element;
typedef ELEMENT_OF(SEISMIC_NORM_WEIGHT) norm_weight_element;
typedef ELEMENT_OF(SEISMIC_NORM_BIAS) norm_bias_element;

#define MERGER_ARGUMENTS                                                                         \
    device const float *hidden [[buffer(SEISMIC_BUFFER_HIDDEN)]],                                \
    device const uchar *norm_weight [[buffer(SEISMIC_BUFFER_NORM_WEIGHT)]],                      \
    device const uchar *norm_bias [[buffer(SEISMIC_BUFFER_NORM_BIAS)]],                          \
    device const uchar *up_weight [[buffer(SEISMIC_BUFFER_UP_WEIGHT)]],                          \
    device const uchar *up_bias [[buffer(SEISMIC_BUFFER_UP_BIAS)]],                              \
    device const uchar *down_weight [[buffer(SEISMIC_BUFFER_DOWN_WEIGHT)]],                      \
    device const uchar *down_bias [[buffer(SEISMIC_BUFFER_DOWN_BIAS)]],                          \
    device float *result [[buffer(SEISMIC_RESULT_0_BUFFER)]],                                    \
    device uchar *normalized [[buffer(SEISMIC_BUFFER_SCRATCH_NORMALIZED)]],                      \
    device uchar *activated [[buffer(SEISMIC_BUFFER_SCRATCH_ACTIVATED)]],                        \
    constant ulong *seismic_words [[buffer(SEISMIC_BUFFER_WORDS)]]

#define NORM_THREADS 256
#define MERGER_WIDTH (SEISMIC_DIM_G * SEISMIC_DIM_H)

kernel void qwen_vision_merger_norm(MERGER_ARGUMENTS,
    uint row [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float partials[NORM_THREADS / 32];
    const uint h = uint(SEISMIC_DIM_H);
    vision::layer_norm<NORM_THREADS, activation, norm_weight_element, norm_bias_element>(hidden + ulong(row) * h,
        normalized + ulong(row) * h * activation::bytes, norm_weight, norm_bias, h,
        as_type<float>(uint(SEISMIC_PARAM_EPSILON)), partials, thread_index, sg, lane);
}

kernel void qwen_vision_merger_up(MERGER_ARGUMENTS,
    uint3 tile [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    PROJECTION_GEMM_SHARED(shared, vision::TILE_M, vision::TILE_N);
    const uint width = uint(MERGER_WIDTH);
    projection::Plain<activation, projection::AllRows> in{normalized, width, 1, width, {}};
    vision::output_bias_gelu<activation, up_bias_element, true> out{activated, width, up_bias};
    auto w = vision::weight<up_packet>(up_weight, SEISMIC_UP_WEIGHT_STRIDE_0, up_element::bytes, width);
    projection::gemm<up_packet, vision::TILE_M, vision::TILE_N>(in, out, w, uint(SEISMIC_DIM_M), width, width,
        tile.y, tile.x, shared, sg, lane);
}

kernel void qwen_vision_merger_down(MERGER_ARGUMENTS,
    uint3 tile [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    PROJECTION_GEMM_SHARED(shared, vision::TILE_M, vision::TILE_N);
    const uint width = uint(MERGER_WIDTH);
    projection::Plain<activation, projection::AllRows> in{activated, width, 1, width, {}};
    vision::output_bias_f32<down_bias_element> out{result, SEISMIC_RESULT_0_STRIDE_0, down_bias, nullptr, 0};
    auto w = vision::weight<down_packet>(down_weight, SEISMIC_DOWN_WEIGHT_STRIDE_0, down_element::bytes, width);
    projection::gemm<down_packet, vision::TILE_M, vision::TILE_N>(in, out, w, uint(SEISMIC_DIM_M),
        uint(SEISMIC_DIM_D), width, tile.y, tile.x, shared, sg, lane);
}
