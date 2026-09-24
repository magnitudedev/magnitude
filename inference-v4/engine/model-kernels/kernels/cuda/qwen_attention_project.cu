// qwen_attention_project: RMS prologue over the F32 hidden rows, then one
// segmented projection q+gate | k | v, each segment with its own
// representation and result. GEMV for M <= 8 (each block forms the A rows in
// shared memory; q8_1 rows staged first for the INT8 candidate), GEMM
// otherwise (A rows or q8_1 rows staged first).
#define SJ_W0 SEISMIC_QUERY_GATE_WEIGHT
#define SJ_W1 SEISMIC_KEY_WEIGHT
#define SJ_W2 SEISMIC_VALUE_WEIGHT
#include "common/projection.cuh"

constexpr bool S8 = SEISMIC_TUNE_INT8 == 1;
using Shape = sj::GemvShape<8, 1, SEISMIC_TUNE_KSPLIT>;
using GShape = sj::GemmShape<SEISMIC_TUNE_BM, 2, 4, 4 - SEISMIC_TUNE_BM / 64>;
using Pro = sj::Rms<SJ_DENSE_KIND(SEISMIC_INPUT_NORM), sj::AllRows>;
using Source = sj::GemvSource<S8, Pro>;
using Out = sj::Store<SJ_DENSE_KIND(SEISMIC_ELEMENT_A)>;

#define PROLOGUE                                                                                          \
    Pro {                                                                                                 \
        reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_HIDDEN)), SEISMIC_HIDDEN_STRIDE_0,      \
            SEISMIC_PTR(SEISMIC_BUFFER_INPUT_NORM), __uint_as_float((unsigned)SEISMIC_PARAM_EPSILON),      \
            SEISMIC_DIM_D, sj::AllRows{}                                                                  \
    }
#define QUERY_ROWS (SEISMIC_DIM_KV * SEISMIC_DIM_G * 2 * SEISMIC_DIM_W)
#define KV_ROWS (SEISMIC_DIM_KV * SEISMIC_DIM_W)
#define STAGING SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_STAGED)
#define GROUPS SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_GROUPS)
#define QUERY_OUT Out{SEISMIC_PTR(SEISMIC_RESULT_0_BUFFER), SEISMIC_RESULT_0_STRIDE_0, 0}
#define KEY_OUT Out{SEISMIC_PTR(SEISMIC_RESULT_1_BUFFER), SEISMIC_RESULT_1_STRIDE_0, 0}
#define VALUE_OUT Out{SEISMIC_PTR(SEISMIC_RESULT_2_BUFFER), SEISMIC_RESULT_2_STRIDE_0, 0}

extern "C" __global__ void qwen_attention_project_stage(SEISMIC_KERNEL_PARAMS) {
    sj::stage_row<S8>(PROLOGUE, blockIdx.x, SEISMIC_DIM_D, STAGING, GROUPS);
}

extern "C" __global__ void qwen_attention_project_gemv(SEISMIC_KERNEL_PARAMS) {
    extern __shared__ uint4 dynamic_shared[];
    __shared__ sj::GemvShared<Shape, Source::type> shared;
    const Source::type x = Source::make(PROLOGUE, reinterpret_cast<sj::u8 *>(dynamic_shared), (unsigned)SEISMIC_DIM_M,
                                        SEISMIC_DIM_D, STAGING, GROUPS);
    const unsigned M = (unsigned)SEISMIC_DIM_M;
    const unsigned long long kblocks = SEISMIC_DIM_D / 64;
    unsigned long long group = Shape::tile_group();
    const unsigned long long groups[3] = {sj::gemv_groups<Shape>(QUERY_ROWS), sj::gemv_groups<Shape>(KV_ROWS),
                                          sj::gemv_groups<Shape>(KV_ROWS)};
    switch (sj::locate_segment(group, groups)) {
    case 0:
        sj::gemv_segment<Shape>(shared, x, M, kblocks, group, QUERY_ROWS,
                                SJ_W0_AT(SEISMIC_PTR(SEISMIC_BUFFER_QUERY_GATE_WEIGHT)), sj::NoWeight{}, QUERY_OUT);
        break;
    case 1:
        sj::gemv_segment<Shape>(shared, x, M, kblocks, group, KV_ROWS,
                                SJ_W1_AT(SEISMIC_PTR(SEISMIC_BUFFER_KEY_WEIGHT)), sj::NoWeight{}, KEY_OUT);
        break;
    case 2:
        sj::gemv_segment<Shape>(shared, x, M, kblocks, group, KV_ROWS,
                                SJ_W2_AT(SEISMIC_PTR(SEISMIC_BUFFER_VALUE_WEIGHT)), sj::NoWeight{}, VALUE_OUT);
        break;
    }
}

extern "C" __global__ void qwen_attention_project_gemm(SEISMIC_KERNEL_PARAMS) {
    extern __shared__ uint4 dynamic_shared[];
    sj::u8 *shared = reinterpret_cast<sj::u8 *>(dynamic_shared);
    const unsigned M = (unsigned)SEISMIC_DIM_M;
    const unsigned long long K = SEISMIC_DIM_D;
    unsigned long long column = blockIdx.x;
    const unsigned long long columns[3] = {sj::gemm_columns(QUERY_ROWS), sj::gemm_columns(KV_ROWS),
                                           sj::gemm_columns(KV_ROWS)};
    switch (sj::locate_segment(column, columns)) {
    case 0:
        sj::gemm_run<GShape, S8>(shared, STAGING, K, GROUPS, M, K, column, QUERY_ROWS,
                                 SJ_W0_AT(SEISMIC_PTR(SEISMIC_BUFFER_QUERY_GATE_WEIGHT)), sj::NoWeight{}, QUERY_OUT);
        break;
    case 1:
        sj::gemm_run<GShape, S8>(shared, STAGING, K, GROUPS, M, K, column, KV_ROWS,
                                 SJ_W1_AT(SEISMIC_PTR(SEISMIC_BUFFER_KEY_WEIGHT)), sj::NoWeight{}, KEY_OUT);
        break;
    case 2:
        sj::gemm_run<GShape, S8>(shared, STAGING, K, GROUPS, M, K, column, KV_ROWS,
                                 SJ_W2_AT(SEISMIC_PTR(SEISMIC_BUFFER_VALUE_WEIGHT)), sj::NoWeight{}, VALUE_OUT);
        break;
    }
}
