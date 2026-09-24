// qwen_attention_output: o_proj of the A-typed gated heads [M, Q * W] with
// the F32 hidden rows added. GEMV for M <= 8 (reading the heads in place, or
// their q8_1 rows for the INT8 candidate), GEMM otherwise (the heads in place,
// or their q8_1 rows), optionally split over K (partials summed in part order
// by the finalize launch).
#define SJ_W0 SEISMIC_OUTPUT_WEIGHT
#include "common/projection.cuh"

constexpr bool S8 = SEISMIC_TUNE_INT8 == 1;
constexpr unsigned SPLIT = SEISMIC_TUNE_SPLIT;
using Shape = sj::GemvShape<8, 1, SEISMIC_TUNE_KSPLIT>;
using GShape = sj::GemmShape<SEISMIC_TUNE_BM, 2, 4, 4 - SEISMIC_TUNE_BM / 64>;
using Pro = sj::Plain<SJ_DENSE_KIND(SEISMIC_ELEMENT_A), sj::AllRows>;
using Epi = sj::ResidualAdd<sj::AllRows>;

#define HEADS (SEISMIC_DIM_Q * SEISMIC_DIM_W)
#define GATED_ROWS Pro{SEISMIC_PTR(SEISMIC_BUFFER_GATED), SEISMIC_GATED_STRIDE_0, sj::AllRows{}}
#define STAGING SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_STAGED)
#define GROUPS SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_GROUPS)
#define OUTPUT SJ_W0_AT(SEISMIC_PTR(SEISMIC_BUFFER_OUTPUT_WEIGHT))
#define PARTIALS reinterpret_cast<float *>(SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_PARTIALS))
#define EPILOGUE                                                                                       \
    Epi {                                                                                              \
        reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_HIDDEN)), SEISMIC_HIDDEN_STRIDE_0,   \
            sj::AllRows{}, reinterpret_cast<float *>(SEISMIC_PTR(SEISMIC_RESULT_0_BUFFER)),            \
            SEISMIC_RESULT_0_STRIDE_0                                                                  \
    }

extern "C" __global__ void qwen_attention_output_stage(SEISMIC_KERNEL_PARAMS) {
    static_assert(!Pro::STAGED, "the 16-bit path reads the heads in place");
    sj::stage_row<S8>(GATED_ROWS, blockIdx.x, HEADS, STAGING, GROUPS);
}

extern "C" __global__ void qwen_attention_output_gemv(SEISMIC_KERNEL_PARAMS) {
    using Source = sj::GemvSource<S8, Pro>;
    __shared__ sj::GemvShared<Shape, Source::type> shared;
    const unsigned long long group = Shape::tile_group();
    if (group < sj::gemv_groups<Shape>(SEISMIC_DIM_D))
        sj::gemv_segment<Shape>(shared, Source::make(GATED_ROWS, nullptr, (unsigned)SEISMIC_DIM_M, HEADS, STAGING, GROUPS),
                                (unsigned)SEISMIC_DIM_M, HEADS / 64, group, SEISMIC_DIM_D, OUTPUT, sj::NoWeight{},
                                EPILOGUE);
}

extern "C" __global__ void qwen_attention_output_gemm(SEISMIC_KERNEL_PARAMS) {
    extern __shared__ uint4 dynamic_shared[];
    if (blockIdx.x >= sj::gemm_columns(SEISMIC_DIM_D))
        return;
    sj::u8 *shared = reinterpret_cast<sj::u8 *>(dynamic_shared);
    const sj::u8 *act = S8 ? STAGING : SEISMIC_PTR(SEISMIC_BUFFER_GATED);
    const unsigned long long stride = S8 ? HEADS : SEISMIC_GATED_STRIDE_0;
    if constexpr (SPLIT > 1)
        sj::gemm_run<GShape, S8>(shared, act, stride, GROUPS, (unsigned)SEISMIC_DIM_M, HEADS, blockIdx.x, SEISMIC_DIM_D,
                                 OUTPUT, sj::NoWeight{},
                                 sj::PartialStore<1>{PARTIALS, SEISMIC_DIM_M, SEISMIC_DIM_D, 0, blockIdx.z}, blockIdx.z,
                                 SPLIT);
    else
        sj::gemm_run<GShape, S8>(shared, act, stride, GROUPS, (unsigned)SEISMIC_DIM_M, HEADS, blockIdx.x, SEISMIC_DIM_D,
                                 OUTPUT, sj::NoWeight{}, EPILOGUE);
}

extern "C" __global__ void qwen_attention_output_finalize(SEISMIC_KERNEL_PARAMS) {
    const Epi epi = EPILOGUE;
    sj::split_finalize<1>(PARTIALS, SPLIT, SEISMIC_DIM_M, SEISMIC_DIM_D,
                          [&](unsigned m, unsigned long long n, float value, float) { epi(m, n, value, 0.0f); });
}
