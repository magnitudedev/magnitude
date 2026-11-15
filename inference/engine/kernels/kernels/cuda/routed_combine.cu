// Grouped combine; the CUDA form of `metal/routed_combine.metal`:
//   stage    the normalized rows' operands: their s8 rows and groups (INT8),
//            or their F16 operand rows (OPERAND);
//   expand   paired GEMM of the shared expert's gate/up over the normalized
//            rows (read in place unless staged) with SiLU . mul into
//            `product` [M, S];
//   quantize each product row's s8 row and groups (INT8);
//   down     shared down GEMM whose epilogue unpermutes the grouped expert
//            outputs in slot order: residual + selected + round_A(shared) * c.
#define KERNEL_W0 SEISMIC_SHARED_DOWN
#define KERNEL_W1 SEISMIC_SHARED_GATE
#define KERNEL_W2 SEISMIC_SHARED_UP
#include "lib/projection/projection.cuh"

using GShape = projection::GemmShape<64, 2, 4, 3>;
using element::Act;
// Whether both GEMMs run on the int8 path (INT8 = 1, which the declaration
// admits where every weight is admitted to it).
#define GROUPED_S8                                                                                         \
    (projection::int8_path<INT8, PACKETS_FORM_INT8(KERNEL_W0), PACKETS_FORM_INT8(KERNEL_W1),               \
                           PACKETS_FORM_INT8(KERNEL_W2)>())

struct CombineEpi {
    element::Out<element::F32> out;
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
    const float *inject;
    projection::u64 inject_stride;
    __device__ __forceinline__ void operator()(unsigned m, projection::u64 n, float value, float) const {
        float selected = 0.0f;
        for (projection::u64 k = 0; k < SEISMIC_DIM_K; ++k) {
            const projection::u64 position = (projection::u64)inverse[m * inverse_stride + k];
            const float projected = element::at<Act>(
                expert_output, (position / tile) * block_stride + (position % tile) * lane_stride + n);
            selected = __fmaf_rn(scores[m * scores_stride + k], projected, selected);
        }
        projection::routed_join<SEISMIC_DIM_SN, (SEISMIC_DIM_IN != 0)>(
            out.offset(m * out_stride), residual + m * residual_stride, inject + m * inject_stride, SEISMIC_DIM_H, n,
            selected, Act::round(value) * coefficient[m]);
    }
};

#ifdef SEISMIC_FORMING_ROUTED_COMBINE_STAGE
template <unsigned INT8, unsigned OPERAND>
__global__ void routed_combine_stage(SEISMIC_KERNEL_PARAMS) {
    using Operands = typename projection::GemmOperands<OPERAND>::type;
    const projection::Plain<Act, projection::AllRows> source{SEISMIC_PTR(SEISMIC_BUFFER_NORMALIZED), SEISMIC_NORMALIZED_STRIDE_0, projection::AllRows{}};
    if constexpr (GROUPED_S8)
        projection::stage_row_s8(source, blockIdx.x, SEISMIC_DIM_H, SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_STAGED),
                                 reinterpret_cast<float2 *>(SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_GROUPS)));
    else
        projection::materialize_operand_row<Operands>(source, blockIdx.x, SEISMIC_DIM_H, SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_STAGED));
}
#endif

#ifdef SEISMIC_FORMING_ROUTED_COMBINE_EXPAND
template <unsigned INT8, unsigned OPERAND>
__global__ void routed_combine_expand(SEISMIC_KERNEL_PARAMS) {
    constexpr bool S8 = GROUPED_S8;
    constexpr bool STAGED = S8 || OPERAND == 1;
    using Operands = typename projection::GemmOperands<OPERAND>::type;
    extern __shared__ uint4 dynamic_shared[];
    if (blockIdx.x >= projection::gemm_columns(SEISMIC_DIM_S))
        return;
    const projection::Glu<typename Operands::Storage, Act> epi{
        element::out_over_scratch<typename Operands::Storage>(SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_PRODUCT)), SEISMIC_DIM_S, 0};
    projection::gemm_run<GShape, S8, Operands>(reinterpret_cast<projection::u8 *>(dynamic_shared),
                             STAGED ? SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_STAGED) : SEISMIC_PTR(SEISMIC_BUFFER_NORMALIZED),
                             STAGED ? SEISMIC_DIM_H : SEISMIC_NORMALIZED_STRIDE_0, SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_GROUPS),
                             (unsigned)SEISMIC_DIM_M, SEISMIC_DIM_H, blockIdx.x, SEISMIC_DIM_S,
                             KERNEL_W1_AT(SEISMIC_PTR(SEISMIC_BUFFER_SHARED_GATE)),
                             KERNEL_W2_AT(SEISMIC_PTR(SEISMIC_BUFFER_SHARED_UP)), epi);
}
#endif

// Each product row's s8 row and groups, the int8 down GEMM's operands (a
// launch of INT8 = 1 alone).
#ifdef SEISMIC_FORMING_ROUTED_COMBINE_QUANTIZE
template <unsigned INT8, unsigned OPERAND>
__global__ void routed_combine_quantize(SEISMIC_KERNEL_PARAMS) {
    projection::stage_row_s8(
        projection::Plain<Act, projection::AllRows>{SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_PRODUCT), SEISMIC_DIM_S,
                                                    projection::AllRows{}},
        blockIdx.x, SEISMIC_DIM_S, SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_PRODUCT_CODES),
        reinterpret_cast<float2 *>(SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_PRODUCT_GROUPS)));
}
#endif

#ifdef SEISMIC_FORMING_ROUTED_COMBINE_DOWN
template <unsigned INT8, unsigned OPERAND>
__global__ void routed_combine_down(SEISMIC_KERNEL_PARAMS) {
    constexpr bool S8 = GROUPED_S8;
    using Operands = typename projection::GemmOperands<OPERAND>::type;
    extern __shared__ uint4 dynamic_shared[];
    if (blockIdx.x >= projection::gemm_columns(SEISMIC_DIM_H))
        return;
    const CombineEpi epi{SEISMIC_RESULT_0_OUT, SEISMIC_RESULT_0_STRIDE_0,
                         reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_RESIDUAL)), SEISMIC_RESIDUAL_STRIDE_0,
                         SEISMIC_PTR(SEISMIC_BUFFER_EXPERT_OUTPUT), SEISMIC_DIM_T, SEISMIC_EXPERT_OUTPUT_STRIDE_0,
                         SEISMIC_EXPERT_OUTPUT_STRIDE_1,
                         reinterpret_cast<const int *>(SEISMIC_PTR(SEISMIC_BUFFER_INVERSE)), SEISMIC_INVERSE_STRIDE_0,
                         reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_SCORES)), SEISMIC_SCORES_STRIDE_0,
                         reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_COEFFICIENT)),
                         reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_INJECT)), SEISMIC_INJECT_STRIDE_0};
    projection::gemm_run<GShape, S8, Operands>(reinterpret_cast<projection::u8 *>(dynamic_shared),
                             S8 ? SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_PRODUCT_CODES) : SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_PRODUCT),
                             SEISMIC_DIM_S, SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_PRODUCT_GROUPS),
                             (unsigned)SEISMIC_DIM_M, SEISMIC_DIM_S, blockIdx.x, SEISMIC_DIM_H,
                             KERNEL_W0_AT(SEISMIC_PTR(SEISMIC_BUFFER_SHARED_DOWN)), projection::NoWeight{}, epi);
}
#endif
