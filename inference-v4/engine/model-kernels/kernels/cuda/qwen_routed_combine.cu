// Grouped combine; the CUDA form of `metal/qwen_routed_combine.metal`:
//   shared   paired GEMM of the shared expert's gate/up over the normalized
//            rows (read in place) with SiLU . mul into `shared_product` [M, S];
//   combine  shared down GEMM whose epilogue unpermutes the grouped expert
//            outputs in slot order: residual + selected + round_A(shared) * c.
#define KERNEL_W0 SEISMIC_SHARED_DOWN
#define KERNEL_W1 SEISMIC_SHARED_GATE
#define KERNEL_W2 SEISMIC_SHARED_UP
#include "common/projection.cuh"

using GShape = projection::GemmShape<64, 2, 4, 3>;
using element::Act;

struct CombineEpi {
    float *out;
    projection::u64 out_stride;
    const float *residual;
    projection::u64 residual_stride;
    const projection::u8 *expert_output;
    projection::u64 tile;
    projection::u64 block_stride;
    projection::u64 lane_stride;
    const int *inverse;
    projection::u64 inverse_stride;
    const float *scores;
    projection::u64 scores_stride;
    const float *coefficient;
    __device__ __forceinline__ void operator()(unsigned m, projection::u64 n, float value, float) const {
        float selected = 0.0f;
        for (projection::u64 k = 0; k < SEISMIC_DIM_K; ++k) {
            const projection::u64 position = (projection::u64)inverse[m * inverse_stride + k];
            const float projected = element::at<Act>(
                expert_output, (position / tile) * block_stride + (position % tile) * lane_stride + n);
            selected = __fmaf_rn(scores[m * scores_stride + k], projected, selected);
        }
        out[m * out_stride + n] = residual[m * residual_stride + n] + selected + Act::round(value) * coefficient[m];
    }
};

extern "C" __global__ void qwen_routed_combine_shared(SEISMIC_KERNEL_PARAMS) {
    extern __shared__ uint4 dynamic_shared[];
    if (blockIdx.x >= projection::gemm_columns(SEISMIC_DIM_S))
        return;
    const projection::SiluMul<Act> epi{SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_SHARED_PRODUCT), SEISMIC_DIM_S};
    projection::gemm_segment<GShape>(reinterpret_cast<projection::u8 *>(dynamic_shared),
                             projection::ActivationRows{SEISMIC_PTR(SEISMIC_BUFFER_NORMALIZED), SEISMIC_NORMALIZED_STRIDE_0},
                             (unsigned)SEISMIC_DIM_M, SEISMIC_DIM_H, blockIdx.x, SEISMIC_DIM_S,
                             KERNEL_W1_AT(SEISMIC_PTR(SEISMIC_BUFFER_SHARED_GATE)),
                             KERNEL_W2_AT(SEISMIC_PTR(SEISMIC_BUFFER_SHARED_UP)), epi);
}

extern "C" __global__ void qwen_routed_combine(SEISMIC_KERNEL_PARAMS) {
    extern __shared__ uint4 dynamic_shared[];
    if (blockIdx.x >= projection::gemm_columns(SEISMIC_DIM_H))
        return;
    const CombineEpi epi{reinterpret_cast<float *>(SEISMIC_PTR(SEISMIC_RESULT_0_BUFFER)), SEISMIC_RESULT_0_STRIDE_0,
                         reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_RESIDUAL)), SEISMIC_RESIDUAL_STRIDE_0,
                         SEISMIC_PTR(SEISMIC_BUFFER_EXPERT_OUTPUT), SEISMIC_DIM_T, SEISMIC_EXPERT_OUTPUT_STRIDE_0,
                         SEISMIC_EXPERT_OUTPUT_STRIDE_1,
                         reinterpret_cast<const int *>(SEISMIC_PTR(SEISMIC_BUFFER_INVERSE)), SEISMIC_INVERSE_STRIDE_0,
                         reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_SCORES)), SEISMIC_SCORES_STRIDE_0,
                         reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_COEFFICIENT))};
    projection::gemm_segment<GShape>(reinterpret_cast<projection::u8 *>(dynamic_shared),
                             projection::ActivationRows{SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_SHARED_PRODUCT), SEISMIC_DIM_S},
                             (unsigned)SEISMIC_DIM_M, SEISMIC_DIM_S, blockIdx.x, SEISMIC_DIM_H,
                             KERNEL_W0_AT(SEISMIC_PTR(SEISMIC_BUFFER_SHARED_DOWN)), projection::NoWeight{}, epi);
}
