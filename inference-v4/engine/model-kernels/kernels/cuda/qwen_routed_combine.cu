// Grouped combine; the CUDA form of `metal/qwen_routed_combine.metal`:
//   shared   paired GEMM of the shared expert's gate/up over the normalized
//            rows (read in place) with SiLU . mul into `shared_product` [M, S];
//   combine  shared down GEMM whose epilogue unpermutes the grouped expert
//            outputs in slot order: residual + selected + round_A(shared) * c.
#define SJ_W0 SEISMIC_SHARED_DOWN
#define SJ_W1 SEISMIC_SHARED_GATE
#define SJ_W2 SEISMIC_SHARED_UP
#include "common/projection.cuh"

using GShape = sj::GemmShape<64, 2, 4, 3>;
constexpr int ACT = SJ_DENSE_KIND(SEISMIC_ELEMENT_A);

struct CombineEpi {
    float *out;
    sj::u64 out_stride;
    const float *residual;
    sj::u64 residual_stride;
    const sj::u8 *expert_output;
    sj::u64 tile;
    sj::u64 block_stride;
    sj::u64 lane_stride;
    const int *inverse;
    sj::u64 inverse_stride;
    const float *scores;
    sj::u64 scores_stride;
    const float *coefficient;
    __device__ __forceinline__ void operator()(unsigned m, sj::u64 n, float value, float) const {
        float selected = 0.0f;
        for (sj::u64 k = 0; k < SEISMIC_DIM_K; ++k) {
            const sj::u64 position = (sj::u64)inverse[m * inverse_stride + k];
            const float projected = sj::Dense<ACT>::load(
                expert_output, (position / tile) * block_stride + (position % tile) * lane_stride + n);
            selected = __fmaf_rn(scores[m * scores_stride + k], projected, selected);
        }
        out[m * out_stride + n] = residual[m * residual_stride + n] + selected + sj::Act::round(value) * coefficient[m];
    }
};

extern "C" __global__ void qwen_routed_combine_shared(SEISMIC_KERNEL_PARAMS) {
    extern __shared__ uint4 dynamic_shared[];
    if (blockIdx.x >= sj::gemm_columns(SEISMIC_DIM_S))
        return;
    const sj::SiluMul<ACT> epi{SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_SHARED_PRODUCT), SEISMIC_DIM_S};
    sj::gemm_segment<GShape>(reinterpret_cast<sj::u8 *>(dynamic_shared), SEISMIC_PTR(SEISMIC_BUFFER_NORMALIZED),
                             SEISMIC_NORMALIZED_STRIDE_0, (unsigned)SEISMIC_DIM_M, SEISMIC_DIM_H, blockIdx.x,
                             SEISMIC_DIM_S,
                             SJ_W1_AT(SEISMIC_PTR(SEISMIC_BUFFER_SHARED_GATE)),
                             SJ_W2_AT(SEISMIC_PTR(SEISMIC_BUFFER_SHARED_UP)), epi);
}

extern "C" __global__ void qwen_routed_combine(SEISMIC_KERNEL_PARAMS) {
    extern __shared__ uint4 dynamic_shared[];
    if (blockIdx.x >= sj::gemm_columns(SEISMIC_DIM_H))
        return;
    const CombineEpi epi{reinterpret_cast<float *>(SEISMIC_PTR(SEISMIC_RESULT_0_BUFFER)), SEISMIC_RESULT_0_STRIDE_0,
                         reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_RESIDUAL)), SEISMIC_RESIDUAL_STRIDE_0,
                         SEISMIC_PTR(SEISMIC_BUFFER_EXPERT_OUTPUT), SEISMIC_DIM_T, SEISMIC_EXPERT_OUTPUT_STRIDE_0,
                         SEISMIC_EXPERT_OUTPUT_STRIDE_1,
                         reinterpret_cast<const int *>(SEISMIC_PTR(SEISMIC_BUFFER_INVERSE)), SEISMIC_INVERSE_STRIDE_0,
                         reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_SCORES)), SEISMIC_SCORES_STRIDE_0,
                         reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_COEFFICIENT))};
    sj::gemm_segment<GShape>(reinterpret_cast<sj::u8 *>(dynamic_shared),
                             SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_SHARED_PRODUCT), SEISMIC_DIM_S,
                             (unsigned)SEISMIC_DIM_M, SEISMIC_DIM_S, blockIdx.x, SEISMIC_DIM_H,
                             SJ_W0_AT(SEISMIC_PTR(SEISMIC_BUFFER_SHARED_DOWN)), sj::NoWeight{}, epi);
}
