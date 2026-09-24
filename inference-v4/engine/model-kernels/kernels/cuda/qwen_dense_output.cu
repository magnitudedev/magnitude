// qwen_dense_output: down projection of the A-typed product rows with the
// F32 residual of their `out_rows` rows added. GEMV for O <= 16 (`gemv` to
// 8 rows, `gemv16` beyond) reading the product in place, GEMM otherwise (the
// product in place, or its q8_1 rows staged first for the INT8 candidate),
// optionally split over K (SPLIT shares into partials, summed in part order
// by the finalize launch).
#define KERNEL_W0 SEISMIC_DOWN_WEIGHT
#include "common/projection.cuh"

constexpr bool S8 = SEISMIC_TUNE_INT8 == 1 && projection::quantizable<packets::W0>;
constexpr unsigned SPLIT = SEISMIC_TUNE_SPLIT;
using GShape = projection::GemmShape<SEISMIC_TUNE_BM, 2, 4, 4 - SEISMIC_TUNE_BM / 64>;
using Pro = projection::Plain<ELEMENT_OF(SEISMIC_ELEMENT_A), projection::AllRows>;
using Epi = projection::Residual<projection::SelectedRows>;

#define PRODUCT Pro{SEISMIC_PTR(SEISMIC_BUFFER_PRODUCT), SEISMIC_PRODUCT_STRIDE_0, projection::AllRows{}}
#define STAGING SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_STAGED)
#define GROUPS SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_GROUPS)
#define DOWN KERNEL_W0_AT(SEISMIC_PTR(SEISMIC_BUFFER_DOWN_WEIGHT))
#define PARTIALS reinterpret_cast<float *>(SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_PARTIALS))
#define EPILOGUE                                                                                       \
    Epi {                                                                                              \
        reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_RESIDUAL)), SEISMIC_RESIDUAL_STRIDE_0, \
            projection::SelectedRows{reinterpret_cast<const int *>(SEISMIC_PTR(SEISMIC_BUFFER_OUT_ROWS))},     \
            reinterpret_cast<float *>(SEISMIC_PTR(SEISMIC_RESULT_0_BUFFER)), SEISMIC_RESULT_0_STRIDE_0 \
    }

// The GEMV over NB column blocks of 8 rows.
template <int NB>
__device__ __forceinline__ void output_gemv(const Pro &product, unsigned M, unsigned long long H, unsigned long long F,
                                            const packets::W0 &down, const Epi &epi) {
    using Shape = projection::GemvShape<8, 1, SEISMIC_TUNE_KSPLIT, NB>;
    __shared__ projection::GemvShared<Shape, Pro> shared;
    const unsigned long long group = Shape::tile_group();
    if (group < projection::gemv_groups<Shape>(H))
        projection::gemv_segment<Shape>(shared, product, M, F / 64, group, H, down, projection::NoWeight{}, epi);
}

extern "C" __global__ void qwen_dense_output_stage(SEISMIC_KERNEL_PARAMS) {
    static_assert(!Pro::STAGED, "the 16-bit path reads the product in place");
    projection::stage_row<S8>(PRODUCT, blockIdx.x, (unsigned)SEISMIC_DIM_O, SEISMIC_DIM_F, STAGING, GROUPS);
}

extern "C" __global__ void qwen_dense_output_gemv(SEISMIC_KERNEL_PARAMS) {
    output_gemv<1>(PRODUCT, (unsigned)SEISMIC_DIM_O, SEISMIC_DIM_H, SEISMIC_DIM_F, DOWN, EPILOGUE);
}

extern "C" __global__ void qwen_dense_output_gemv16(SEISMIC_KERNEL_PARAMS) {
    output_gemv<2>(PRODUCT, (unsigned)SEISMIC_DIM_O, SEISMIC_DIM_H, SEISMIC_DIM_F, DOWN, EPILOGUE);
}

extern "C" __global__ void qwen_dense_output_gemm(SEISMIC_KERNEL_PARAMS) {
    extern __shared__ uint4 dynamic_shared[];
    if (blockIdx.x >= projection::gemm_columns(SEISMIC_DIM_H))
        return;
    projection::u8 *shared = reinterpret_cast<projection::u8 *>(dynamic_shared);
    const projection::u8 *act = S8 ? STAGING : SEISMIC_PTR(SEISMIC_BUFFER_PRODUCT);
    const unsigned long long stride = S8 ? SEISMIC_DIM_F : SEISMIC_PRODUCT_STRIDE_0;
    if constexpr (SPLIT > 1)
        projection::gemm_run<GShape, S8>(shared, act, stride, GROUPS, (unsigned)SEISMIC_DIM_O, SEISMIC_DIM_F, blockIdx.x,
                                 SEISMIC_DIM_H, DOWN, projection::NoWeight{},
                                 projection::PartialStore<1>{PARTIALS, SEISMIC_DIM_O, SEISMIC_DIM_H, 0, blockIdx.z}, blockIdx.z,
                                 SPLIT);
    else
        projection::gemm_run<GShape, S8>(shared, act, stride, GROUPS, (unsigned)SEISMIC_DIM_O, SEISMIC_DIM_F, blockIdx.x,
                                 SEISMIC_DIM_H, DOWN, projection::NoWeight{}, EPILOGUE);
}

extern "C" __global__ void qwen_dense_output_finalize(SEISMIC_KERNEL_PARAMS) {
    const Epi epi = EPILOGUE;
    projection::split_finalize<1>(PARTIALS, SPLIT, SEISMIC_DIM_O, SEISMIC_DIM_H,
                          [&](unsigned m, unsigned long long n, float value, float) { epi(m, n, value, 0.0f); });
}
