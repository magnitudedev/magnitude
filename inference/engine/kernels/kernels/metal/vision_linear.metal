// vision_linear: one GEMM tile of y = x . transpose(weight) on the projection
// library with the `output_linear` epilogue (lib/vision/vision.h): the bias,
// the clamp, then the activation, the gate or the F32 residual.
#include "lib/vision/vision.h"

#if defined(SEISMIC_ELEMENT_A_REPRESENTATION_BF16)
typedef element::Bf16 activation;
#elif defined(SEISMIC_ELEMENT_A_REPRESENTATION_F16)
typedef element::F16 activation;
#else
#error "vision_linear requires a bf16 or f16 activation"
#endif

typedef vision::dense_packet<ELEMENT_KIND(SEISMIC_WEIGHT)>::type weight_packet;
typedef ELEMENT_OF(SEISMIC_WEIGHT) weight_element;
typedef ELEMENT_OF(SEISMIC_BIAS) bias_element;
typedef ELEMENT_OF(SEISMIC_ELEMENT_Y) output_element;

kernel void vision_linear(device const uchar *x [[buffer(SEISMIC_BUFFER_X)]],
    device const uchar *weight [[buffer(SEISMIC_BUFFER_WEIGHT)]],
    device const uchar *bias [[buffer(SEISMIC_BUFFER_BIAS)]],
    device const float *residual [[buffer(SEISMIC_BUFFER_RESIDUAL)]],
    device const uchar *gate [[buffer(SEISMIC_BUFFER_GATE)]],
    device const float *minimum [[buffer(SEISMIC_BUFFER_MINIMUM)]],
    device const float *maximum [[buffer(SEISMIC_BUFFER_MAXIMUM)]],
    device uchar *result [[buffer(SEISMIC_RESULT_0_BUFFER)]],
    constant ulong *seismic_words [[buffer(SEISMIC_BUFFER_WORDS)]],
    uint3 tile [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    PROJECTION_GEMM_SHARED(shared, vision::TILE_M, vision::TILE_N);
    const uint n = uint(SEISMIC_DIM_N), k = uint(SEISMIC_DIM_K);
    constexpr bool clamped = SEISMIC_DIM_NC == 1;
    projection::Plain<activation, projection::AllRows> in{x, SEISMIC_X_STRIDE_0, 1, k, {}};
    vision::output_linear<activation, bias_element, output_element, SEISMIC_DIM_NB == 1, SEISMIC_DIM_NR == 1,
        SEISMIC_DIM_NG == 1, clamped> out{result, SEISMIC_RESULT_0_STRIDE_0, bias, residual,
        SEISMIC_RESIDUAL_STRIDE_1, gate, SEISMIC_GATE_STRIDE_1, clamped ? minimum[0] : 0.0f,
        clamped ? maximum[0] : 0.0f, int(SEISMIC_PARAM_ACTIVATION)};
    auto w = vision::weight<weight_packet>(weight, SEISMIC_WEIGHT_STRIDE_0, weight_element::bytes, k);
    projection::gemm<weight_packet, vision::TILE_M, vision::TILE_N>(in, out, w, uint(SEISMIC_DIM_M), n, k, tile.y,
        tile.x, shared, sg, lane);
}
