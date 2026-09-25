// dense_output: down projection of the A-typed product rows with the
// F32 residual of their `out_rows` rows added. GEMV for O <= 16 (`gemv` to
// 8 rows, `gemv16` beyond) reading the product in place, GEMM otherwise:
// `gemm_small` to 64 rows (the product in place, or the INT8 candidate's q8_1
// rows staged by `stage_s8`), `gemm` beyond (the product in place); up to
// SPLIT_ROWS rows split over K into SPLIT shares of partials that `finalize`
// sums in part order.
#define KERNEL_W0 SEISMIC_DOWN_WEIGHT
#include "lib/projection/projection.cuh"

using Pro = projection::Plain<ELEMENT_OF(SEISMIC_ELEMENT_A), projection::AllRows>;
using Epi = projection::Residual<projection::SelectedRows>;

#define PRODUCT Pro{SEISMIC_PTR(SEISMIC_BUFFER_PRODUCT), SEISMIC_PRODUCT_STRIDE_0, projection::AllRows{}}
#define STAGING SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_STAGED)
#define GROUPS SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_GROUPS)
#define DOWN KERNEL_W0_AT(SEISMIC_PTR(SEISMIC_BUFFER_DOWN_WEIGHT))
#define PARTIALS reinterpret_cast<float *>(SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_PARTIALS))
#define EPILOGUE                                                                                          \
    Epi {                                                                                                 \
        reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_RESIDUAL)), SEISMIC_RESIDUAL_STRIDE_0,  \
            projection::SelectedRows{reinterpret_cast<const int *>(SEISMIC_PTR(SEISMIC_BUFFER_OUT_ROWS))}, \
            reinterpret_cast<float *>(SEISMIC_PTR(SEISMIC_RESULT_0_BUFFER)), SEISMIC_RESULT_0_STRIDE_0    \
    }

// The GEMV over NB column blocks of 8 rows.
template <int NB, int KSPLIT>
__device__ __forceinline__ void output_gemv(const Pro &product, unsigned M, unsigned long long H, unsigned long long F,
                                            const packets::W0 &down, const Epi &epi) {
    using Shape = projection::GemvShape<8, 1, KSPLIT, NB>;
    __shared__ projection::GemvShared<Shape, Pro> shared;
    const unsigned long long group = Shape::tile_group();
    if (group < projection::gemv_groups<Shape>(H))
        projection::gemv_segment<Shape>(shared, product, M, F / 64, group, H, down, projection::NoWeight{}, epi);
}

#ifdef SEISMIC_FORMING_DENSE_OUTPUT_STAGE_S8
template <unsigned INT8>
__global__ void dense_output_stage_s8(SEISMIC_KERNEL_PARAMS) {
    constexpr bool S8 = INT8 == 1 && projection::quantizable<packets::W0>;
    projection::stage_row<S8>(PRODUCT, blockIdx.x, SEISMIC_DIM_F, STAGING, GROUPS);
}
#endif

#ifdef SEISMIC_FORMING_DENSE_OUTPUT_GEMV
template <unsigned KSPLIT>
__global__ void dense_output_gemv(SEISMIC_KERNEL_PARAMS) {
    output_gemv<1, KSPLIT>(PRODUCT, (unsigned)SEISMIC_DIM_O, SEISMIC_DIM_H, SEISMIC_DIM_F, DOWN, EPILOGUE);
}
#endif

#ifdef SEISMIC_FORMING_DENSE_OUTPUT_GEMV16
template <unsigned KSPLIT>
__global__ void dense_output_gemv16(SEISMIC_KERNEL_PARAMS) {
    output_gemv<2, KSPLIT>(PRODUCT, (unsigned)SEISMIC_DIM_O, SEISMIC_DIM_H, SEISMIC_DIM_F, DOWN, EPILOGUE);
}
#endif

#ifdef SEISMIC_FORMING_DENSE_OUTPUT_GEMM_SMALL
template <unsigned INT8>
__global__ void dense_output_gemm_small(SEISMIC_KERNEL_PARAMS) {
    constexpr bool S8 = INT8 == 1 && projection::quantizable<packets::W0>;
    extern __shared__ uint4 dynamic_shared[];
    if (blockIdx.x < projection::gemm_columns(SEISMIC_DIM_H))
        projection::gemm_run_split<projection::SmallGemm, S8>(
            reinterpret_cast<projection::u8 *>(dynamic_shared), S8 ? STAGING : SEISMIC_PTR(SEISMIC_BUFFER_PRODUCT),
            S8 ? SEISMIC_DIM_F : SEISMIC_PRODUCT_STRIDE_0, GROUPS, (unsigned)SEISMIC_DIM_O, SEISMIC_DIM_F, blockIdx.x,
            SEISMIC_DIM_H, DOWN, EPILOGUE, PARTIALS);
}
#endif

#ifdef SEISMIC_FORMING_DENSE_OUTPUT_GEMM
extern "C" __global__ void dense_output_gemm(SEISMIC_KERNEL_PARAMS) {
    extern __shared__ uint4 dynamic_shared[];
    if (blockIdx.x < projection::gemm_columns(SEISMIC_DIM_H))
        projection::gemm_run_split<projection::LargeGemm, false>(
            reinterpret_cast<projection::u8 *>(dynamic_shared), SEISMIC_PTR(SEISMIC_BUFFER_PRODUCT),
            SEISMIC_PRODUCT_STRIDE_0, GROUPS, (unsigned)SEISMIC_DIM_O, SEISMIC_DIM_F, blockIdx.x, SEISMIC_DIM_H, DOWN,
            EPILOGUE, PARTIALS);
}
#endif

#ifdef SEISMIC_FORMING_DENSE_OUTPUT_FINALIZE
extern "C" __global__ void dense_output_finalize(SEISMIC_KERNEL_PARAMS) {
    const Epi epi = EPILOGUE;
    projection::split_finalize<1>(PARTIALS, projection::SPLIT, SEISMIC_DIM_O, SEISMIC_DIM_H,
                                  [&](unsigned m, unsigned long long n, float value, float) { epi(m, n, value, 0.0f); });
}
#endif
