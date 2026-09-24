// attention_output: o_proj of the A-typed gated heads [M, Q * W] with
// the F32 hidden rows added. GEMV for M <= 16 (`gemv` to 8 rows, `gemv16`
// beyond) reading the heads in place, GEMM otherwise: `gemm_small` to 64 rows
// (the heads in place, or the INT8 candidate's q8_1 rows staged by
// `stage_s8`), `gemm` beyond (the heads in place); up to SPLIT_ROWS rows split
// over K into SPLIT shares of partials that `finalize` sums in part order.
#define KERNEL_W0 SEISMIC_OUTPUT_WEIGHT
#include "common/projection.cuh"

constexpr bool S8 = SEISMIC_TUNE_INT8 == 1 && projection::quantizable<packets::W0>;
using Pro = projection::Plain<ELEMENT_OF(SEISMIC_ELEMENT_A), projection::AllRows>;
using Epi = projection::Residual<projection::AllRows>;

#define HEADS (SEISMIC_DIM_Q * SEISMIC_DIM_W)
#define GATED_ROWS Pro{SEISMIC_PTR(SEISMIC_BUFFER_GATED), SEISMIC_GATED_STRIDE_0, projection::AllRows{}}
#define STAGING SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_STAGED)
#define GROUPS SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_GROUPS)
#define OUTPUT KERNEL_W0_AT(SEISMIC_PTR(SEISMIC_BUFFER_OUTPUT_WEIGHT))
#define PARTIALS reinterpret_cast<float *>(SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_PARTIALS))
#define EPILOGUE                                                                                     \
    Epi {                                                                                            \
        reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_HIDDEN)), SEISMIC_HIDDEN_STRIDE_0, \
            projection::AllRows{}, reinterpret_cast<float *>(SEISMIC_PTR(SEISMIC_RESULT_0_BUFFER)),  \
            SEISMIC_RESULT_0_STRIDE_0                                                                \
    }

// The GEMV over NB column blocks of 8 rows.
template <int NB>
__device__ __forceinline__ void output_gemv(const Pro &heads, unsigned M, unsigned long long K, unsigned long long D,
                                            const packets::W0 &output, const Epi &epi) {
    using Shape = projection::GemvShape<8, 1, SEISMIC_TUNE_KSPLIT, NB>;
    __shared__ projection::GemvShared<Shape, Pro> shared;
    const unsigned long long group = Shape::tile_group();
    if (group < projection::gemv_groups<Shape>(D))
        projection::gemv_segment<Shape>(shared, heads, M, K / 64, group, D, output, projection::NoWeight{}, epi);
}

extern "C" __global__ void attention_output_stage_s8(SEISMIC_KERNEL_PARAMS) {
    static_assert(!Pro::STAGED, "the 16-bit path reads the heads in place");
    projection::stage_row<S8>(GATED_ROWS, blockIdx.x, HEADS, STAGING, GROUPS);
}

extern "C" __global__ void attention_output_gemv(SEISMIC_KERNEL_PARAMS) {
    output_gemv<1>(GATED_ROWS, (unsigned)SEISMIC_DIM_M, HEADS, SEISMIC_DIM_D, OUTPUT, EPILOGUE);
}

extern "C" __global__ void attention_output_gemv16(SEISMIC_KERNEL_PARAMS) {
    output_gemv<2>(GATED_ROWS, (unsigned)SEISMIC_DIM_M, HEADS, SEISMIC_DIM_D, OUTPUT, EPILOGUE);
}

extern "C" __global__ void attention_output_gemm_small(SEISMIC_KERNEL_PARAMS) {
    extern __shared__ uint4 dynamic_shared[];
    if (blockIdx.x < projection::gemm_columns(SEISMIC_DIM_D))
        projection::gemm_run_split<projection::SmallGemm, S8>(
            reinterpret_cast<projection::u8 *>(dynamic_shared), S8 ? STAGING : SEISMIC_PTR(SEISMIC_BUFFER_GATED),
            S8 ? HEADS : SEISMIC_GATED_STRIDE_0, GROUPS, (unsigned)SEISMIC_DIM_M, HEADS, blockIdx.x, SEISMIC_DIM_D,
            OUTPUT, EPILOGUE, PARTIALS);
}

extern "C" __global__ void attention_output_gemm(SEISMIC_KERNEL_PARAMS) {
    extern __shared__ uint4 dynamic_shared[];
    if (blockIdx.x < projection::gemm_columns(SEISMIC_DIM_D))
        projection::gemm_run_split<projection::LargeGemm, false>(
            reinterpret_cast<projection::u8 *>(dynamic_shared), SEISMIC_PTR(SEISMIC_BUFFER_GATED),
            SEISMIC_GATED_STRIDE_0, GROUPS, (unsigned)SEISMIC_DIM_M, HEADS, blockIdx.x, SEISMIC_DIM_D, OUTPUT,
            EPILOGUE, PARTIALS);
}

extern "C" __global__ void attention_output_finalize(SEISMIC_KERNEL_PARAMS) {
    const Epi epi = EPILOGUE;
    projection::split_finalize<1>(PARTIALS, projection::SPLIT, SEISMIC_DIM_M, SEISMIC_DIM_D,
                                  [&](unsigned m, unsigned long long n, float value, float) { epi(m, n, value, 0.0f); });
}
