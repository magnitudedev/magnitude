// head_logits_rows: the draft head's vocabulary projection of already
// normalized A feature rows into F32 logits (a K1 projection without a
// prologue). GEMV for O <= 8 (reading the features in place, or their q8_1
// rows for the INT8 candidate), GEMM otherwise (reading the features in
// place, or their q8_1 rows staged first).
#define SJ_W0 SEISMIC_WEIGHT
#include "common/projection.cuh"

constexpr bool S8 = SEISMIC_TUNE_INT8 == 1;
using Shape = sj::GemvShape<8, 1, SEISMIC_TUNE_KSPLIT>;
using GShape = sj::GemmShape<SEISMIC_TUNE_BM, 2, 4, 4 - SEISMIC_TUNE_BM / 64>;
using Pro = sj::Plain<SJ_DENSE_KIND(SEISMIC_ELEMENT_A), sj::AllRows>;
using Epi = sj::Store<0>;

#define FEATURES Pro{SEISMIC_PTR(SEISMIC_BUFFER_FEATURES), SEISMIC_FEATURES_STRIDE_0, sj::AllRows{}}
#define STAGING SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_STAGED)
#define GROUPS SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_GROUPS)
#define HEAD SJ_W0_AT(SEISMIC_PTR(SEISMIC_BUFFER_WEIGHT))
#define EPILOGUE Epi{SEISMIC_PTR(SEISMIC_RESULT_0_BUFFER), SEISMIC_RESULT_0_STRIDE_0, 0}

extern "C" __global__ void head_logits_rows_stage(SEISMIC_KERNEL_PARAMS) {
    static_assert(!Pro::STAGED, "the 16-bit path reads the features in place");
    sj::stage_row<S8>(FEATURES, blockIdx.x, SEISMIC_DIM_D, STAGING, GROUPS);
}

extern "C" __global__ void head_logits_rows_gemv(SEISMIC_KERNEL_PARAMS) {
    using Source = sj::GemvSource<S8, Pro>;
    __shared__ sj::GemvShared<Shape, Source::type> shared;
    const unsigned long long group = Shape::tile_group();
    if (group < sj::gemv_groups<Shape>(SEISMIC_DIM_V))
        sj::gemv_segment<Shape>(shared, Source::make(FEATURES, nullptr, (unsigned)SEISMIC_DIM_O, SEISMIC_DIM_D, STAGING, GROUPS),
                                (unsigned)SEISMIC_DIM_O, SEISMIC_DIM_D / 64, group, SEISMIC_DIM_V, HEAD, sj::NoWeight{},
                                EPILOGUE);
}

extern "C" __global__ void head_logits_rows_gemm(SEISMIC_KERNEL_PARAMS) {
    extern __shared__ uint4 dynamic_shared[];
    if (blockIdx.x >= sj::gemm_columns(SEISMIC_DIM_V))
        return;
    const sj::u8 *act = S8 ? STAGING : SEISMIC_PTR(SEISMIC_BUFFER_FEATURES);
    const unsigned long long stride = S8 ? SEISMIC_DIM_D : SEISMIC_FEATURES_STRIDE_0;
    sj::gemm_run<GShape, S8>(reinterpret_cast<sj::u8 *>(dynamic_shared), act, stride, GROUPS, (unsigned)SEISMIC_DIM_O,
                             SEISMIC_DIM_D, blockIdx.x, SEISMIC_DIM_V, HEAD, sj::NoWeight{}, EPILOGUE);
}
