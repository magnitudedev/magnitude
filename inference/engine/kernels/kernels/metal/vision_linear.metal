// vision_linear: one GEMM tile of y = x . transpose(weight) on the projection
// library with the `output_linear` epilogue (lib/vision/vision.h): the bias,
// the clamp, then the activation, the gate or the F32 residual.
#define KERNEL_A SEISMIC_ELEMENT_A
#include "lib/vision/vision.h"

#if ELEMENT_KIND(KERNEL_A) != ELEMENT_BF16 && ELEMENT_KIND(KERNEL_A) != ELEMENT_F16
#error "vision_linear requires a bf16 or f16 activation"
#endif
typedef activation::Act activation_element;

typedef ELEMENT_OF(SEISMIC_WEIGHT) weight_element;
typedef packets::Dense<weight_element> weight_packet;
typedef ELEMENT_OF(SEISMIC_BIAS) bias_element;
typedef ELEMENT_OF(SEISMIC_ELEMENT_Y) output_element;

kernel void vision_linear(device const uchar *x [[buffer(SEISMIC_BUFFER_X)]],
    device const uchar *weight [[buffer(SEISMIC_BUFFER_WEIGHT)]],
    device const uchar *bias [[buffer(SEISMIC_BUFFER_BIAS)]],
    device const float *residual [[buffer(SEISMIC_BUFFER_RESIDUAL)]],
    device const uchar *gate [[buffer(SEISMIC_BUFFER_GATE)]],
    device const float *minimum [[buffer(SEISMIC_BUFFER_MINIMUM)]],
    device const float *maximum [[buffer(SEISMIC_BUFFER_MAXIMUM)]],
    SEISMIC_RESULT_0_PARAM(result),
    constant ulong *seismic_words [[buffer(SEISMIC_BUFFER_WORDS)]],
    uint3 tile [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    PROJECTION_GEMM_SHARED(shared, vision::TILE_M, vision::TILE_N);
    const uint n = uint(SEISMIC_DIM_N), k = uint(SEISMIC_DIM_K);
    constexpr bool clamped = SEISMIC_DIM_NC == 1;
    projection::Plain<activation_element, projection::AllRows> in{x, SEISMIC_X_STRIDE_0, 1, k, {}};
    vision::output_linear<activation_element, bias_element, output_element, SEISMIC_DIM_NB == 1, SEISMIC_DIM_NR == 1,
        SEISMIC_DIM_NG == 1, clamped> out{element::Out<output_element>(result), SEISMIC_RESULT_0_STRIDE_0, bias, residual,
        SEISMIC_RESIDUAL_STRIDE_1, gate, SEISMIC_GATE_STRIDE_1, clamped ? minimum[0] : 0.0f,
        clamped ? maximum[0] : 0.0f, int(SEISMIC_PARAM_ACTIVATION)};
    auto w = vision::weight<weight_element>(weight, SEISMIC_WEIGHT_STRIDE_0, k);
    projection::gemm<weight_packet, vision::TILE_M, vision::TILE_N>(in, out, w, uint(SEISMIC_DIM_M), n, k, tile.y,
        tile.x, shared, sg, lane);
}
