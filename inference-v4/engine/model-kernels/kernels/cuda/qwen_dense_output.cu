// qwen_dense_output: down projection of the A-typed product rows with the
// F32 residual of their `out_rows` rows added. GEMV for O <= 8 (reading the
// product in place, or its q8_1 rows for the INT8 candidate), GEMM otherwise
// (the product in place, or its q8_1 rows), optionally split over K (SPLIT
// shares into partials, summed in part order by the finalize launch).
#define SJ_W0 SEISMIC_DOWN_WEIGHT
#include "common/projection.cuh"

constexpr bool S8 = SEISMIC_TUNE_INT8 == 1;
constexpr unsigned SPLIT = SEISMIC_TUNE_SPLIT;
using Shape = sj::GemvShape<8, 1, SEISMIC_TUNE_KSPLIT>;
using GShape = sj::GemmShape<SEISMIC_TUNE_BM, 2, 4, 4 - SEISMIC_TUNE_BM / 64>;
using Pro = sj::Plain<SJ_DENSE_KIND(SEISMIC_ELEMENT_A), sj::AllRows>;
using Epi = sj::ResidualAdd<sj::SelectedRows>;

#define PRODUCT Pro{SEISMIC_PTR(SEISMIC_BUFFER_PRODUCT), SEISMIC_PRODUCT_STRIDE_0, sj::AllRows{}}
#define STAGING SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_STAGED)
#define GROUPS SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_GROUPS)
#define DOWN SJ_W0_AT(SEISMIC_PTR(SEISMIC_BUFFER_DOWN_WEIGHT))
#define PARTIALS reinterpret_cast<float *>(SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_PARTIALS))
#define EPILOGUE                                                                                       \
    Epi {                                                                                              \
        reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_RESIDUAL)), SEISMIC_RESIDUAL_STRIDE_0, \
            sj::SelectedRows{reinterpret_cast<const int *>(SEISMIC_PTR(SEISMIC_BUFFER_OUT_ROWS))},     \
            reinterpret_cast<float *>(SEISMIC_PTR(SEISMIC_RESULT_0_BUFFER)), SEISMIC_RESULT_0_STRIDE_0 \
    }

extern "C" __global__ void qwen_dense_output_stage(SEISMIC_KERNEL_PARAMS) {
    static_assert(!Pro::STAGED, "the 16-bit path reads the product in place");
    sj::stage_row<S8>(PRODUCT, blockIdx.x, SEISMIC_DIM_F, STAGING, GROUPS);
}

extern "C" __global__ void qwen_dense_output_gemv(SEISMIC_KERNEL_PARAMS) {
    using Source = sj::GemvSource<S8, Pro>;
    __shared__ sj::GemvShared<Shape, Source::type> shared;
    const unsigned long long group = Shape::tile_group();
    if (group < sj::gemv_groups<Shape>(SEISMIC_DIM_H))
        sj::gemv_segment<Shape>(shared, Source::make(PRODUCT, nullptr, (unsigned)SEISMIC_DIM_O, SEISMIC_DIM_F, STAGING, GROUPS),
                                (unsigned)SEISMIC_DIM_O, SEISMIC_DIM_F / 64, group, SEISMIC_DIM_H, DOWN, sj::NoWeight{},
                                EPILOGUE);
}

extern "C" __global__ void qwen_dense_output_gemm(SEISMIC_KERNEL_PARAMS) {
    extern __shared__ uint4 dynamic_shared[];
    if (blockIdx.x >= sj::gemm_columns(SEISMIC_DIM_H))
        return;
    sj::u8 *shared = reinterpret_cast<sj::u8 *>(dynamic_shared);
    const sj::u8 *act = S8 ? STAGING : SEISMIC_PTR(SEISMIC_BUFFER_PRODUCT);
    const unsigned long long stride = S8 ? SEISMIC_DIM_F : SEISMIC_PRODUCT_STRIDE_0;
    if constexpr (SPLIT > 1)
        sj::gemm_run<GShape, S8>(shared, act, stride, GROUPS, (unsigned)SEISMIC_DIM_O, SEISMIC_DIM_F, blockIdx.x,
                                 SEISMIC_DIM_H, DOWN, sj::NoWeight{},
                                 sj::PartialStore<1>{PARTIALS, SEISMIC_DIM_O, SEISMIC_DIM_H, 0, blockIdx.z}, blockIdx.z,
                                 SPLIT);
    else
        sj::gemm_run<GShape, S8>(shared, act, stride, GROUPS, (unsigned)SEISMIC_DIM_O, SEISMIC_DIM_F, blockIdx.x,
                                 SEISMIC_DIM_H, DOWN, sj::NoWeight{}, EPILOGUE);
}

extern "C" __global__ void qwen_dense_output_finalize(SEISMIC_KERNEL_PARAMS) {
    const Epi epi = EPILOGUE;
    sj::split_finalize<1>(PARTIALS, SPLIT, SEISMIC_DIM_O, SEISMIC_DIM_H,
                          [&](unsigned m, unsigned long long n, float value, float) { epi(m, n, value, 0.0f); });
}
