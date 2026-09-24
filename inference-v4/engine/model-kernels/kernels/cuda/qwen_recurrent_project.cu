// qwen_recurrent_project: RMS prologue over the F32 hidden rows, then one
// segmented projection qkv | z | alpha | beta into the columns of the A-typed
// result, each segment with its own representation. GEMV for M <= 8 (each
// block forms the A rows in shared memory; q8_1 rows staged first for the
// INT8 candidate), GEMM otherwise (A rows or q8_1 rows staged first).
#define SJ_W0 SEISMIC_QKV_WEIGHT
#define SJ_W1 SEISMIC_GATE_WEIGHT
#define SJ_W2 SEISMIC_ALPHA_WEIGHT
#define SJ_W3 SEISMIC_BETA_WEIGHT
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
            SEISMIC_DIM_H, sj::AllRows{}                                                                  \
    }
#define QKV_ROWS ((2 * SEISMIC_DIM_NK + SEISMIC_DIM_NV) * SEISMIC_DIM_W)
#define Z_ROWS (SEISMIC_DIM_NV * SEISMIC_DIM_W)
#define GATE_ROWS SEISMIC_DIM_NV
#define STAGING SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_STAGED)
#define GROUPS SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_GROUPS)
#define OUT(offset) Out{SEISMIC_PTR(SEISMIC_RESULT_0_BUFFER), SEISMIC_RESULT_0_STRIDE_0, (offset)}

extern "C" __global__ void qwen_recurrent_project_stage(SEISMIC_KERNEL_PARAMS) {
    sj::stage_row<S8>(PROLOGUE, blockIdx.x, SEISMIC_DIM_H, STAGING, GROUPS);
}

extern "C" __global__ void qwen_recurrent_project_gemv(SEISMIC_KERNEL_PARAMS) {
    extern __shared__ uint4 dynamic_shared[];
    __shared__ sj::GemvShared<Shape, Source::type> shared;
    const Source::type x = Source::make(PROLOGUE, reinterpret_cast<sj::u8 *>(dynamic_shared), (unsigned)SEISMIC_DIM_M,
                                        SEISMIC_DIM_H, STAGING, GROUPS);
    const unsigned M = (unsigned)SEISMIC_DIM_M;
    const unsigned long long kblocks = SEISMIC_DIM_H / 64;
    unsigned long long group = Shape::tile_group();
    const unsigned long long groups[4] = {sj::gemv_groups<Shape>(QKV_ROWS), sj::gemv_groups<Shape>(Z_ROWS),
                                          sj::gemv_groups<Shape>(GATE_ROWS), sj::gemv_groups<Shape>(GATE_ROWS)};
    switch (sj::locate_segment(group, groups)) {
    case 0:
        sj::gemv_segment<Shape>(shared, x, M, kblocks, group, QKV_ROWS, SJ_W0_AT(SEISMIC_PTR(SEISMIC_BUFFER_QKV_WEIGHT)),
                                sj::NoWeight{}, OUT(0));
        break;
    case 1:
        sj::gemv_segment<Shape>(shared, x, M, kblocks, group, Z_ROWS, SJ_W1_AT(SEISMIC_PTR(SEISMIC_BUFFER_GATE_WEIGHT)),
                                sj::NoWeight{}, OUT(QKV_ROWS));
        break;
    case 2:
        sj::gemv_segment<Shape>(shared, x, M, kblocks, group, GATE_ROWS,
                                SJ_W2_AT(SEISMIC_PTR(SEISMIC_BUFFER_ALPHA_WEIGHT)), sj::NoWeight{},
                                OUT(QKV_ROWS + Z_ROWS));
        break;
    case 3:
        sj::gemv_segment<Shape>(shared, x, M, kblocks, group, GATE_ROWS,
                                SJ_W3_AT(SEISMIC_PTR(SEISMIC_BUFFER_BETA_WEIGHT)), sj::NoWeight{},
                                OUT(QKV_ROWS + Z_ROWS + GATE_ROWS));
        break;
    }
}

extern "C" __global__ void qwen_recurrent_project_gemm(SEISMIC_KERNEL_PARAMS) {
    extern __shared__ uint4 dynamic_shared[];
    sj::u8 *shared = reinterpret_cast<sj::u8 *>(dynamic_shared);
    const unsigned M = (unsigned)SEISMIC_DIM_M;
    const unsigned long long K = SEISMIC_DIM_H;
    unsigned long long column = blockIdx.x;
    const unsigned long long columns[4] = {sj::gemm_columns(QKV_ROWS), sj::gemm_columns(Z_ROWS),
                                           sj::gemm_columns(GATE_ROWS), sj::gemm_columns(GATE_ROWS)};
    switch (sj::locate_segment(column, columns)) {
    case 0:
        sj::gemm_run<GShape, S8>(shared, STAGING, K, GROUPS, M, K, column, QKV_ROWS,
                                 SJ_W0_AT(SEISMIC_PTR(SEISMIC_BUFFER_QKV_WEIGHT)), sj::NoWeight{}, OUT(0));
        break;
    case 1:
        sj::gemm_run<GShape, S8>(shared, STAGING, K, GROUPS, M, K, column, Z_ROWS,
                                 SJ_W1_AT(SEISMIC_PTR(SEISMIC_BUFFER_GATE_WEIGHT)), sj::NoWeight{}, OUT(QKV_ROWS));
        break;
    case 2:
        sj::gemm_run<GShape, S8>(shared, STAGING, K, GROUPS, M, K, column, GATE_ROWS,
                                 SJ_W2_AT(SEISMIC_PTR(SEISMIC_BUFFER_ALPHA_WEIGHT)), sj::NoWeight{},
                                 OUT(QKV_ROWS + Z_ROWS));
        break;
    case 3:
        sj::gemm_run<GShape, S8>(shared, STAGING, K, GROUPS, M, K, column, GATE_ROWS,
                                 SJ_W3_AT(SEISMIC_PTR(SEISMIC_BUFFER_BETA_WEIGHT)), sj::NoWeight{},
                                 OUT(QKV_ROWS + Z_ROWS + GATE_ROWS));
        break;
    }
}
