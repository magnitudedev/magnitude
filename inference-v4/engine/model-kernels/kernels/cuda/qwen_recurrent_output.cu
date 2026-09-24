// qwen_recurrent_output: the gated per-head RMS prologue
//   gated = round_A(round_A(mixed * rsqrt(sum_head mixed^2 / W + eps) * norm)
//                   * round_A(silu(z)))
// with z the gate columns of the recurrent projection, then the ssm_out
// projection with the F32 hidden rows added. GEMV for M <= 8 (each block
// forms the A rows in shared memory; q8_1 rows staged first for the INT8
// candidate), GEMM otherwise (A rows or q8_1 rows staged first), optionally
// split over K (partials summed in part order by the finalize).
#define SJ_W0 SEISMIC_OUTPUT_WEIGHT
#include "common/projection.cuh"

constexpr bool S8 = SEISMIC_TUNE_INT8 == 1;
constexpr unsigned SPLIT = SEISMIC_TUNE_SPLIT;
using Shape = sj::GemvShape<8, 1, SEISMIC_TUNE_KSPLIT>;
using GShape = sj::GemmShape<SEISMIC_TUNE_BM, 2, 4, 4 - SEISMIC_TUNE_BM / 64>;
using Pro = sj::GatedRms<SJ_DENSE_KIND(SEISMIC_ELEMENT_A), SJ_DENSE_KIND(SEISMIC_ELEMENT_A),
                         SJ_DENSE_KIND(SEISMIC_RECURRENT_NORM), (unsigned)SEISMIC_DIM_W, (unsigned)SEISMIC_DIM_NV,
                         sj::AllRows>;
using Source = sj::GemvSource<S8, Pro>;
using Epi = sj::ResidualAdd<sj::AllRows>;

#define GATED (SEISMIC_DIM_NV * SEISMIC_DIM_W)
// z: the gate segment of the projection row, after qkv.
#define PROLOGUE                                                                                             \
    Pro {                                                                                                    \
        SEISMIC_PTR(SEISMIC_BUFFER_MIXED), SEISMIC_MIXED_STRIDE_0,                                           \
            SEISMIC_PTR(SEISMIC_BUFFER_PROJECTION) +                                                         \
                (2 * SEISMIC_DIM_NK + SEISMIC_DIM_NV) * SEISMIC_DIM_W * sizeof(unsigned short),              \
            SEISMIC_PROJECTION_STRIDE_0, SEISMIC_PTR(SEISMIC_BUFFER_RECURRENT_NORM),                         \
            __uint_as_float((unsigned)SEISMIC_PARAM_EPSILON), sj::AllRows{}                                  \
    }
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

extern "C" __global__ void qwen_recurrent_output_stage(SEISMIC_KERNEL_PARAMS) {
    sj::stage_row<S8>(PROLOGUE, blockIdx.x, GATED, STAGING, GROUPS);
}

extern "C" __global__ void qwen_recurrent_output_gemv(SEISMIC_KERNEL_PARAMS) {
    extern __shared__ uint4 dynamic_shared[];
    __shared__ sj::GemvShared<Shape, Source::type> shared;
    const Source::type x = Source::make(PROLOGUE, reinterpret_cast<sj::u8 *>(dynamic_shared), (unsigned)SEISMIC_DIM_M,
                                        GATED, STAGING, GROUPS);
    const unsigned long long group = Shape::tile_group();
    if (group < sj::gemv_groups<Shape>(SEISMIC_DIM_H))
        sj::gemv_segment<Shape>(shared, x, (unsigned)SEISMIC_DIM_M, GATED / 64, group, SEISMIC_DIM_H, OUTPUT,
                                sj::NoWeight{}, EPILOGUE);
}

extern "C" __global__ void qwen_recurrent_output_gemm(SEISMIC_KERNEL_PARAMS) {
    extern __shared__ uint4 dynamic_shared[];
    if (blockIdx.x >= sj::gemm_columns(SEISMIC_DIM_H))
        return;
    sj::u8 *shared = reinterpret_cast<sj::u8 *>(dynamic_shared);
    if constexpr (SPLIT > 1)
        sj::gemm_run<GShape, S8>(shared, STAGING, GATED, GROUPS, (unsigned)SEISMIC_DIM_M, GATED, blockIdx.x,
                                 SEISMIC_DIM_H, OUTPUT, sj::NoWeight{},
                                 sj::PartialStore<1>{PARTIALS, SEISMIC_DIM_M, SEISMIC_DIM_H, 0, blockIdx.z}, blockIdx.z,
                                 SPLIT);
    else
        sj::gemm_run<GShape, S8>(shared, STAGING, GATED, GROUPS, (unsigned)SEISMIC_DIM_M, GATED, blockIdx.x,
                                 SEISMIC_DIM_H, OUTPUT, sj::NoWeight{}, EPILOGUE);
}

extern "C" __global__ void qwen_recurrent_output_finalize(SEISMIC_KERNEL_PARAMS) {
    const Epi epi = EPILOGUE;
    sj::split_finalize<1>(PARTIALS, SPLIT, SEISMIC_DIM_M, SEISMIC_DIM_H,
                          [&](unsigned m, unsigned long long n, float value, float) { epi(m, n, value, 0.0f); });
}
