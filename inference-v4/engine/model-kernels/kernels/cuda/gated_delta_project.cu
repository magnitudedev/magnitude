// gated_delta_project: RMS prologue over the F32 hidden rows, then one
// segmented projection qkv | z | alpha | beta into the columns of the A-typed
// result, each segment with its own representation. GEMV for M <= 16 (`gemv`
// to 8 rows, `gemv16` beyond; at M = 1 the block forms the A row in shared
// memory, else `stage` forms the A rows first), GEMM otherwise: `gemm_small`
// to 64 rows (A rows, or the INT8 candidate's q8_1 rows staged by
// `stage_s8`), `gemm` beyond (A rows).
#define KERNEL_W0 SEISMIC_QKV_WEIGHT
#define KERNEL_W1 SEISMIC_GATE_WEIGHT
#define KERNEL_W2 SEISMIC_ALPHA_WEIGHT
#define KERNEL_W3 SEISMIC_BETA_WEIGHT
#include "lib/projection/projection.cuh"

constexpr bool S8 =
    SEISMIC_TUNE_INT8 == 1 && projection::quantizable<packets::W0, packets::W1, packets::W2, packets::W3>;
using Pro = projection::Rms<ELEMENT_OF(SEISMIC_INPUT_NORM), projection::AllRows>;
using Source = projection::GemvSource<Pro>;
using Out = projection::Store<ELEMENT_OF(SEISMIC_ELEMENT_A)>;

#define PROLOGUE                                                                                          \
    Pro {                                                                                                 \
        reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_HIDDEN)), SEISMIC_HIDDEN_STRIDE_0,      \
            SEISMIC_PTR(SEISMIC_BUFFER_INPUT_NORM), __uint_as_float((unsigned)SEISMIC_PARAM_EPSILON),      \
            SEISMIC_DIM_H, projection::AllRows{}                                                                  \
    }
#define QKV_ROWS ((2 * SEISMIC_DIM_NK + SEISMIC_DIM_NV) * SEISMIC_DIM_W)
#define Z_ROWS (SEISMIC_DIM_NV * SEISMIC_DIM_W)
#define GATE_ROWS SEISMIC_DIM_NV
#define STAGING SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_STAGED)
#define GROUPS SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_GROUPS)
#define OUT(offset) Out{SEISMIC_PTR(SEISMIC_RESULT_0_BUFFER), SEISMIC_RESULT_0_STRIDE_0, (offset)}

// The segmented GEMV over NB column blocks of 8 rows: segment rows `rows`,
// weights `w0..w3`, epilogues `out` (each placing its segment's columns).
template <int NB>
__device__ __forceinline__ void project_gemv(const Pro &pro, projection::u8 *row, const projection::u8 *staged,
                                             unsigned M, unsigned long long H, const unsigned long long (&rows)[4],
                                             const packets::W0 &w0, const packets::W1 &w1, const packets::W2 &w2,
                                             const packets::W3 &w3, const Out (&out)[4]) {
    using Shape = projection::GemvShape<8, 1, SEISMIC_TUNE_KSPLIT, NB>;
    __shared__ projection::GemvShared<Shape, Source::type> shared;
    const Source::type x = Source::make(pro, row, M, H, staged);
    const unsigned long long kblocks = H / 64;
    unsigned long long group = Shape::tile_group();
    const unsigned long long groups[4] = {projection::gemv_groups<Shape>(rows[0]), projection::gemv_groups<Shape>(rows[1]),
                                          projection::gemv_groups<Shape>(rows[2]), projection::gemv_groups<Shape>(rows[3])};
    switch (projection::locate_segment(group, groups)) {
    case 0:
        projection::gemv_segment<Shape>(shared, x, M, kblocks, group, rows[0], w0, projection::NoWeight{}, out[0]);
        break;
    case 1:
        projection::gemv_segment<Shape>(shared, x, M, kblocks, group, rows[1], w1, projection::NoWeight{}, out[1]);
        break;
    case 2:
        projection::gemv_segment<Shape>(shared, x, M, kblocks, group, rows[2], w2, projection::NoWeight{}, out[2]);
        break;
    case 3:
        projection::gemv_segment<Shape>(shared, x, M, kblocks, group, rows[3], w3, projection::NoWeight{}, out[3]);
        break;
    }
}

#define SEGMENT_ROWS {QKV_ROWS, Z_ROWS, GATE_ROWS, GATE_ROWS}
#define SEGMENT_OUT {OUT(0), OUT(QKV_ROWS), OUT(QKV_ROWS + Z_ROWS), OUT(QKV_ROWS + Z_ROWS + GATE_ROWS)}
#define WEIGHTS                                                                                               \
    KERNEL_W0_AT(SEISMIC_PTR(SEISMIC_BUFFER_QKV_WEIGHT)), KERNEL_W1_AT(SEISMIC_PTR(SEISMIC_BUFFER_GATE_WEIGHT)), \
        KERNEL_W2_AT(SEISMIC_PTR(SEISMIC_BUFFER_ALPHA_WEIGHT)), KERNEL_W3_AT(SEISMIC_PTR(SEISMIC_BUFFER_BETA_WEIGHT))

// The segmented GEMM of one row band over the staged rows (A rows, or q8_1
// rows with Q): block column blockIdx.x of qkv | z | alpha | beta.
template <class Shape, bool Q>
__device__ __forceinline__ void project_gemm(const projection::u8 *staged, const void *groups, unsigned M,
                                             unsigned long long H, const unsigned long long (&rows)[4],
                                             const packets::W0 &w0, const packets::W1 &w1, const packets::W2 &w2,
                                             const packets::W3 &w3, const Out (&out)[4]) {
    extern __shared__ uint4 dynamic_shared[];
    projection::u8 *shared = reinterpret_cast<projection::u8 *>(dynamic_shared);
    unsigned long long column = blockIdx.x;
    const unsigned long long columns[4] = {projection::gemm_columns(rows[0]), projection::gemm_columns(rows[1]),
                                           projection::gemm_columns(rows[2]), projection::gemm_columns(rows[3])};
    switch (projection::locate_segment(column, columns)) {
    case 0:
        projection::gemm_run<Shape, Q>(shared, staged, H, groups, M, H, column, rows[0], w0, projection::NoWeight{},
                                       out[0]);
        break;
    case 1:
        projection::gemm_run<Shape, Q>(shared, staged, H, groups, M, H, column, rows[1], w1, projection::NoWeight{},
                                       out[1]);
        break;
    case 2:
        projection::gemm_run<Shape, Q>(shared, staged, H, groups, M, H, column, rows[2], w2, projection::NoWeight{},
                                       out[2]);
        break;
    case 3:
        projection::gemm_run<Shape, Q>(shared, staged, H, groups, M, H, column, rows[3], w3, projection::NoWeight{},
                                       out[3]);
        break;
    }
}

extern "C" __global__ void gated_delta_project_stage(SEISMIC_KERNEL_PARAMS) {
    projection::stage_row<false>(PROLOGUE, blockIdx.x, SEISMIC_DIM_H, STAGING, GROUPS);
}

extern "C" __global__ void gated_delta_project_stage_s8(SEISMIC_KERNEL_PARAMS) {
    projection::stage_row<S8>(PROLOGUE, blockIdx.x, SEISMIC_DIM_H, STAGING, GROUPS);
}

extern "C" __global__ void gated_delta_project_gemv(SEISMIC_KERNEL_PARAMS) {
    extern __shared__ uint4 dynamic_shared[];
    const unsigned long long rows[4] = SEGMENT_ROWS;
    const Out out[4] = SEGMENT_OUT;
    project_gemv<1>(PROLOGUE, reinterpret_cast<projection::u8 *>(dynamic_shared), STAGING, (unsigned)SEISMIC_DIM_M,
                    SEISMIC_DIM_H, rows, WEIGHTS, out);
}

extern "C" __global__ void gated_delta_project_gemv16(SEISMIC_KERNEL_PARAMS) {
    const unsigned long long rows[4] = SEGMENT_ROWS;
    const Out out[4] = SEGMENT_OUT;
    project_gemv<2>(PROLOGUE, nullptr, STAGING, (unsigned)SEISMIC_DIM_M, SEISMIC_DIM_H, rows, WEIGHTS, out);
}

extern "C" __global__ void gated_delta_project_gemm_small(SEISMIC_KERNEL_PARAMS) {
    const unsigned long long rows[4] = SEGMENT_ROWS;
    const Out out[4] = SEGMENT_OUT;
    project_gemm<projection::SmallGemm, S8>(STAGING, GROUPS, (unsigned)SEISMIC_DIM_M, SEISMIC_DIM_H, rows, WEIGHTS,
                                            out);
}

extern "C" __global__ void gated_delta_project_gemm(SEISMIC_KERNEL_PARAMS) {
    const unsigned long long rows[4] = SEGMENT_ROWS;
    const Out out[4] = SEGMENT_OUT;
    project_gemm<projection::LargeGemm, false>(STAGING, GROUPS, (unsigned)SEISMIC_DIM_M, SEISMIC_DIM_H, rows, WEIGHTS,
                                               out);
}
