// head_logits_rows: the draft head's vocabulary projection of already
// normalized A feature rows into F32 logits (a K1 projection without a
// prologue). GEMV for O <= 16 (`gemv` to 8 rows, `gemv16` beyond) reading
// the features in place, GEMM otherwise (reading the features in place, or
// their q8_1 rows staged first for the INT8 candidate).
#define KERNEL_W0 SEISMIC_WEIGHT
#include "common/projection.cuh"

constexpr bool S8 = SEISMIC_TUNE_INT8 == 1 && projection::quantizable<packets::W0>;
using GShape = projection::GemmShape<SEISMIC_TUNE_BM, 2, 4, 4 - SEISMIC_TUNE_BM / 64>;
using Pro = projection::Plain<ELEMENT_OF(SEISMIC_ELEMENT_A), projection::AllRows>;
using Epi = projection::Store<element::F32>;

#define FEATURES Pro{SEISMIC_PTR(SEISMIC_BUFFER_FEATURES), SEISMIC_FEATURES_STRIDE_0, projection::AllRows{}}
#define STAGING SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_STAGED)
#define GROUPS SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_GROUPS)
#define HEAD KERNEL_W0_AT(SEISMIC_PTR(SEISMIC_BUFFER_WEIGHT))
#define EPILOGUE Epi{SEISMIC_PTR(SEISMIC_RESULT_0_BUFFER), SEISMIC_RESULT_0_STRIDE_0, 0}

// The GEMV over NB column blocks of 8 rows.
template <int NB>
__device__ __forceinline__ void logits_gemv(const Pro &features, unsigned O, unsigned long long D,
                                            unsigned long long V, const packets::W0 &head, const Epi &epi) {
    using Shape = projection::GemvShape<8, 1, SEISMIC_TUNE_KSPLIT, NB>;
    __shared__ projection::GemvShared<Shape, Pro> shared;
    const unsigned long long group = Shape::tile_group();
    if (group < projection::gemv_groups<Shape>(V))
        projection::gemv_segment<Shape>(shared, features, O, D / 64, group, V, head, projection::NoWeight{}, epi);
}

extern "C" __global__ void head_logits_rows_stage(SEISMIC_KERNEL_PARAMS) {
    static_assert(!Pro::STAGED, "the 16-bit path reads the features in place");
    projection::stage_row<S8>(FEATURES, blockIdx.x, (unsigned)SEISMIC_DIM_O, SEISMIC_DIM_D, STAGING, GROUPS);
}

extern "C" __global__ void head_logits_rows_gemv(SEISMIC_KERNEL_PARAMS) {
    logits_gemv<1>(FEATURES, (unsigned)SEISMIC_DIM_O, SEISMIC_DIM_D, SEISMIC_DIM_V, HEAD, EPILOGUE);
}

extern "C" __global__ void head_logits_rows_gemv16(SEISMIC_KERNEL_PARAMS) {
    logits_gemv<2>(FEATURES, (unsigned)SEISMIC_DIM_O, SEISMIC_DIM_D, SEISMIC_DIM_V, HEAD, EPILOGUE);
}

extern "C" __global__ void head_logits_rows_gemm(SEISMIC_KERNEL_PARAMS) {
    extern __shared__ uint4 dynamic_shared[];
    if (blockIdx.x >= projection::gemm_columns(SEISMIC_DIM_V))
        return;
    const projection::u8 *act = S8 ? STAGING : SEISMIC_PTR(SEISMIC_BUFFER_FEATURES);
    const unsigned long long stride = S8 ? SEISMIC_DIM_D : SEISMIC_FEATURES_STRIDE_0;
    projection::gemm_run<GShape, S8>(reinterpret_cast<projection::u8 *>(dynamic_shared), act, stride, GROUPS, (unsigned)SEISMIC_DIM_O,
                             SEISMIC_DIM_D, blockIdx.x, SEISMIC_DIM_V, HEAD, projection::NoWeight{}, EPILOGUE);
}
