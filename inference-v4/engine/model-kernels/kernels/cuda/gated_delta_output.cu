// gated_delta_output: the gated per-head RMS prologue
//   gated = round_A(round_A(mixed * rsqrt(sum_head mixed^2 / W + eps) * norm)
//                   * round_A(silu(z)))
// with z the gate columns of the recurrent projection, then the ssm_out
// projection with the F32 hidden rows added. GEMV for M <= 16 (`gemv` to 8
// rows, `gemv16` beyond; at M = 1 the block forms the A row in shared memory,
// else `stage` forms the A rows first), GEMM otherwise: `gemm_small` to 64
// rows (A rows, or the INT8 candidate's q8_1 rows staged by `stage_s8`),
// `gemm` beyond (A rows); up to SPLIT_ROWS rows split over K into SPLIT shares
// of partials that `finalize` sums in part order.
#define KERNEL_W0 SEISMIC_OUTPUT_WEIGHT
#include "lib/projection/projection.cuh"

using Pro = projection::GatedRms<ELEMENT_OF(SEISMIC_ELEMENT_A), ELEMENT_OF(SEISMIC_ELEMENT_A),
                         ELEMENT_OF(SEISMIC_RECURRENT_NORM), (unsigned)SEISMIC_DIM_W, (unsigned)SEISMIC_DIM_NV,
                         projection::AllRows>;
using Source = projection::GemvSource<Pro>;
using Epi = projection::Residual<projection::AllRows>;

#define GATED (SEISMIC_DIM_NV * SEISMIC_DIM_W)
// z: the gate segment of the projection row, after qkv.
#define PROLOGUE                                                                                             \
    Pro {                                                                                                    \
        SEISMIC_PTR(SEISMIC_BUFFER_MIXED), SEISMIC_MIXED_STRIDE_0,                                           \
            SEISMIC_PTR(SEISMIC_BUFFER_PROJECTION) +                                                         \
                (2 * SEISMIC_DIM_NK + SEISMIC_DIM_NV) * SEISMIC_DIM_W * sizeof(unsigned short),              \
            SEISMIC_PROJECTION_STRIDE_0, SEISMIC_PTR(SEISMIC_BUFFER_RECURRENT_NORM),                         \
            __uint_as_float((unsigned)SEISMIC_PARAM_EPSILON), projection::AllRows{}                                  \
    }
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
template <int NB, int KSPLIT>
__device__ __forceinline__ void output_gemv(const Pro &pro, projection::u8 *row, const projection::u8 *staged,
                                            unsigned M, unsigned long long K, unsigned long long H,
                                            const packets::W0 &output, const Epi &epi) {
    using Shape = projection::GemvShape<8, 1, KSPLIT, NB>;
    __shared__ projection::GemvShared<Shape, Source::type> shared;
    const Source::type x = Source::make(pro, row, M, K, staged);
    const unsigned long long group = Shape::tile_group();
    if (group < projection::gemv_groups<Shape>(H))
        projection::gemv_segment<Shape>(shared, x, M, K / 64, group, H, output, projection::NoWeight{}, epi);
}

#ifdef SEISMIC_FORMING_GATED_DELTA_OUTPUT_STAGE
template <unsigned INT8>
__global__ void gated_delta_output_stage(SEISMIC_KERNEL_PARAMS) {
    projection::stage_row<false>(PROLOGUE, blockIdx.x, GATED, STAGING, GROUPS);
}
#endif

#ifdef SEISMIC_FORMING_GATED_DELTA_OUTPUT_STAGE_S8
template <unsigned INT8>
__global__ void gated_delta_output_stage_s8(SEISMIC_KERNEL_PARAMS) {
    constexpr bool S8 = INT8 == 1 && projection::quantizable<packets::W0>;
    projection::stage_row<S8>(PROLOGUE, blockIdx.x, GATED, STAGING, GROUPS);
}
#endif

#ifdef SEISMIC_FORMING_GATED_DELTA_OUTPUT_GEMV
template <unsigned KSPLIT>
__global__ void gated_delta_output_gemv(SEISMIC_KERNEL_PARAMS) {
    extern __shared__ uint4 dynamic_shared[];
    output_gemv<1, KSPLIT>(PROLOGUE, reinterpret_cast<projection::u8 *>(dynamic_shared), STAGING, (unsigned)SEISMIC_DIM_M,
                   GATED, SEISMIC_DIM_H, OUTPUT, EPILOGUE);
}
#endif

#ifdef SEISMIC_FORMING_GATED_DELTA_OUTPUT_GEMV16
template <unsigned KSPLIT>
__global__ void gated_delta_output_gemv16(SEISMIC_KERNEL_PARAMS) {
    output_gemv<2, KSPLIT>(PROLOGUE, nullptr, STAGING, (unsigned)SEISMIC_DIM_M, GATED, SEISMIC_DIM_H, OUTPUT, EPILOGUE);
}
#endif

#ifdef SEISMIC_FORMING_GATED_DELTA_OUTPUT_GEMM_SMALL
template <unsigned INT8>
__global__ void gated_delta_output_gemm_small(SEISMIC_KERNEL_PARAMS) {
    constexpr bool S8 = INT8 == 1 && projection::quantizable<packets::W0>;
    extern __shared__ uint4 dynamic_shared[];
    if (blockIdx.x < projection::gemm_columns(SEISMIC_DIM_H))
        projection::gemm_run_split<projection::SmallGemm, S8>(reinterpret_cast<projection::u8 *>(dynamic_shared),
                                                              STAGING, GATED, GROUPS, (unsigned)SEISMIC_DIM_M, GATED,
                                                              blockIdx.x, SEISMIC_DIM_H, OUTPUT, EPILOGUE, PARTIALS);
}
#endif

#ifdef SEISMIC_FORMING_GATED_DELTA_OUTPUT_GEMM
extern "C" __global__ void gated_delta_output_gemm(SEISMIC_KERNEL_PARAMS) {
    extern __shared__ uint4 dynamic_shared[];
    if (blockIdx.x < projection::gemm_columns(SEISMIC_DIM_H))
        projection::gemm_run_split<projection::LargeGemm, false>(reinterpret_cast<projection::u8 *>(dynamic_shared),
                                                                 STAGING, GATED, GROUPS, (unsigned)SEISMIC_DIM_M,
                                                                 GATED, blockIdx.x, SEISMIC_DIM_H, OUTPUT, EPILOGUE,
                                                                 PARTIALS);
}
#endif

#ifdef SEISMIC_FORMING_GATED_DELTA_OUTPUT_FINALIZE
extern "C" __global__ void gated_delta_output_finalize(SEISMIC_KERNEL_PARAMS) {
    const Epi epi = EPILOGUE;
    projection::split_finalize<1>(PARTIALS, projection::SPLIT, SEISMIC_DIM_M, SEISMIC_DIM_H,
                                  [&](unsigned m, unsigned long long n, float value, float) { epi(m, n, value, 0.0f); });
}
#endif
