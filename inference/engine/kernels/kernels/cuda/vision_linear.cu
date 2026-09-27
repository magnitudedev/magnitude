// vision_linear: one block tile of y = x . transpose(weight) on the
// dense-weight tensor-core GEMM with the `Linear` epilogue
// (lib/vision/vision.cuh): the bias, the clamp, then the activation, the gate
// or the F32 residual.
#include "lib/vision/vision.cuh"

using vision::u32;
typedef element::Act A;
typedef ELEMENT_OF(SEISMIC_WEIGHT) Weight;
typedef ELEMENT_OF(SEISMIC_BIAS) BiasElement;
typedef ELEMENT_OF(SEISMIC_ELEMENT_Y) Y;

extern "C" __global__ void vision_linear(SEISMIC_KERNEL_PARAMS) {
    __shared__ vision::GemmShared shared;
    constexpr bool clamped = SEISMIC_DIM_NC == 1;
    const float *minimum = reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_MINIMUM));
    const float *maximum = reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_MAXIMUM));
    const vision::Linear<A, BiasElement, Y, SEISMIC_DIM_NB == 1, SEISMIC_DIM_NR == 1, SEISMIC_DIM_NG == 1, clamped>
        epilogue{SEISMIC_PTR(SEISMIC_RESULT_0_BUFFER),
                 SEISMIC_RESULT_0_STRIDE_0,
                 SEISMIC_PTR(SEISMIC_BUFFER_BIAS),
                 reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_RESIDUAL)),
                 SEISMIC_RESIDUAL_STRIDE_1,
                 SEISMIC_PTR(SEISMIC_BUFFER_GATE),
                 SEISMIC_GATE_STRIDE_1,
                 clamped ? minimum[0] : 0.0f,
                 clamped ? maximum[0] : 0.0f,
                 (int)SEISMIC_PARAM_ACTIVATION};
    vision::gemm<A, Weight>(shared, SEISMIC_PTR(SEISMIC_BUFFER_X), SEISMIC_X_STRIDE_0, SEISMIC_PTR(SEISMIC_BUFFER_WEIGHT),
                            (u32)SEISMIC_DIM_M, (u32)SEISMIC_DIM_N, (u32)SEISMIC_DIM_K, epilogue);
}
