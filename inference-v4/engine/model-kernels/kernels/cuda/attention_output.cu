// attention_output: o_proj of the A-typed gated heads [M, Q * W] with
// the F32 hidden rows added. GEMV for M <= 16 (`gemv` to 8 rows, `gemv16`
// beyond) reading the heads in place, GEMM otherwise (the heads in place, or
// their q8_1 rows staged first for the INT8 candidate), optionally split over
// K (partials summed in part order by the finalize launch).
#define KERNEL_W0 SEISMIC_OUTPUT_WEIGHT
#include "common/projection.cuh"

constexpr bool S8 = SEISMIC_TUNE_INT8 == 1 && projection::quantizable<packets::W0>;
constexpr unsigned SPLIT = SEISMIC_TUNE_SPLIT;
using GShape = projection::GemmShape<SEISMIC_TUNE_BM, 2, 4, 4 - SEISMIC_TUNE_BM / 64>;
using Pro = projection::Plain<ELEMENT_OF(SEISMIC_ELEMENT_A), projection::AllRows>;
using Epi = projection::Residual<projection::AllRows>;

#define HEADS (SEISMIC_DIM_Q * SEISMIC_DIM_W)
#define GATED_ROWS Pro{SEISMIC_PTR(SEISMIC_BUFFER_GATED), SEISMIC_GATED_STRIDE_0, projection::AllRows{}}
#define STAGING SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_STAGED)
#define GROUPS SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_GROUPS)
#define OUTPUT KERNEL_W0_AT(SEISMIC_PTR(SEISMIC_BUFFER_OUTPUT_WEIGHT))
#define PARTIALS reinterpret_cast<float *>(SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_PARTIALS))
#define EPILOGUE                                                                                       \
    Epi {                                                                                              \
        reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_HIDDEN)), SEISMIC_HIDDEN_STRIDE_0,   \
            projection::AllRows{}, reinterpret_cast<float *>(SEISMIC_PTR(SEISMIC_RESULT_0_BUFFER)),            \
            SEISMIC_RESULT_0_STRIDE_0                                                                  \
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

extern "C" __global__ void attention_output_stage(SEISMIC_KERNEL_PARAMS) {
    static_assert(!Pro::STAGED, "the 16-bit path reads the heads in place");
    projection::stage_row<S8>(GATED_ROWS, blockIdx.x, (unsigned)SEISMIC_DIM_M, HEADS, STAGING, GROUPS);
}

extern "C" __global__ void attention_output_gemv(SEISMIC_KERNEL_PARAMS) {
    output_gemv<1>(GATED_ROWS, (unsigned)SEISMIC_DIM_M, HEADS, SEISMIC_DIM_D, OUTPUT, EPILOGUE);
}

extern "C" __global__ void attention_output_gemv16(SEISMIC_KERNEL_PARAMS) {
    output_gemv<2>(GATED_ROWS, (unsigned)SEISMIC_DIM_M, HEADS, SEISMIC_DIM_D, OUTPUT, EPILOGUE);
}

extern "C" __global__ void attention_output_gemm(SEISMIC_KERNEL_PARAMS) {
    extern __shared__ uint4 dynamic_shared[];
    if (blockIdx.x >= projection::gemm_columns(SEISMIC_DIM_D))
        return;
    projection::u8 *shared = reinterpret_cast<projection::u8 *>(dynamic_shared);
    const projection::u8 *act = S8 ? STAGING : SEISMIC_PTR(SEISMIC_BUFFER_GATED);
    const unsigned long long stride = S8 ? HEADS : SEISMIC_GATED_STRIDE_0;
    if constexpr (SPLIT > 1)
        projection::gemm_run<GShape, S8>(shared, act, stride, GROUPS, (unsigned)SEISMIC_DIM_M, HEADS, blockIdx.x, SEISMIC_DIM_D,
                                 OUTPUT, projection::NoWeight{},
                                 projection::PartialStore<1>{PARTIALS, SEISMIC_DIM_M, SEISMIC_DIM_D, 0, blockIdx.z}, blockIdx.z,
                                 SPLIT);
    else
        projection::gemm_run<GShape, S8>(shared, act, stride, GROUPS, (unsigned)SEISMIC_DIM_M, HEADS, blockIdx.x, SEISMIC_DIM_D,
                                 OUTPUT, projection::NoWeight{}, EPILOGUE);
}

extern "C" __global__ void attention_output_finalize(SEISMIC_KERNEL_PARAMS) {
    const Epi epi = EPILOGUE;
    projection::split_finalize<1>(PARTIALS, SPLIT, SEISMIC_DIM_M, SEISMIC_DIM_D,
                          [&](unsigned m, unsigned long long n, float value, float) { epi(m, n, value, 0.0f); });
}
