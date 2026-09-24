// qwen_attention_project: RMS prologue over the F32 hidden rows, then one
// segmented projection q+gate | k | v, each segment with its own
// representation and result. GEMV for M <= 16 (`gemv` to 8 rows, `gemv16`
// beyond; at M = 1 the block forms the A row in shared memory, else the
// staging launch forms the A rows first), GEMM otherwise (A rows, or q8_1 rows
// for the INT8 candidate, staged first).
#define KERNEL_W0 SEISMIC_QUERY_GATE_WEIGHT
#define KERNEL_W1 SEISMIC_KEY_WEIGHT
#define KERNEL_W2 SEISMIC_VALUE_WEIGHT
#include "common/projection.cuh"

constexpr bool S8 = SEISMIC_TUNE_INT8 == 1 && projection::quantizable<packets::W0, packets::W1, packets::W2>;
using GShape = projection::GemmShape<SEISMIC_TUNE_BM, 2, 4, 4 - SEISMIC_TUNE_BM / 64>;
using Pro = projection::Rms<ELEMENT_OF(SEISMIC_INPUT_NORM), projection::AllRows>;
using Source = projection::GemvSource<Pro>;
using Out = projection::Store<ELEMENT_OF(SEISMIC_ELEMENT_A)>;

#define PROLOGUE                                                                                          \
    Pro {                                                                                                 \
        reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_HIDDEN)), SEISMIC_HIDDEN_STRIDE_0,      \
            SEISMIC_PTR(SEISMIC_BUFFER_INPUT_NORM), __uint_as_float((unsigned)SEISMIC_PARAM_EPSILON),      \
            SEISMIC_DIM_D, projection::AllRows{}                                                                  \
    }
#define QUERY_ROWS (SEISMIC_DIM_KV * SEISMIC_DIM_G * 2 * SEISMIC_DIM_W)
#define KV_ROWS (SEISMIC_DIM_KV * SEISMIC_DIM_W)
#define STAGING SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_STAGED)
#define GROUPS SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_GROUPS)
#define QUERY_OUT Out{SEISMIC_PTR(SEISMIC_RESULT_0_BUFFER), SEISMIC_RESULT_0_STRIDE_0, 0}
#define KEY_OUT Out{SEISMIC_PTR(SEISMIC_RESULT_1_BUFFER), SEISMIC_RESULT_1_STRIDE_0, 0}
#define VALUE_OUT Out{SEISMIC_PTR(SEISMIC_RESULT_2_BUFFER), SEISMIC_RESULT_2_STRIDE_0, 0}

// The segmented GEMV over NB column blocks of 8 rows: query+gate | key | value.
template <int NB>
__device__ __forceinline__ void project_gemv(const Pro &pro, projection::u8 *row, const projection::u8 *staged,
                                             unsigned M, unsigned long long D, unsigned long long query_rows,
                                             unsigned long long kv_rows, const packets::W0 &query,
                                             const packets::W1 &key, const packets::W2 &value, const Out &query_out,
                                             const Out &key_out, const Out &value_out) {
    using Shape = projection::GemvShape<8, 1, SEISMIC_TUNE_KSPLIT, NB>;
    __shared__ projection::GemvShared<Shape, Source::type> shared;
    const Source::type x = Source::make(pro, row, M, D, staged);
    const unsigned long long kblocks = D / 64;
    unsigned long long group = Shape::tile_group();
    const unsigned long long groups[3] = {projection::gemv_groups<Shape>(query_rows),
                                          projection::gemv_groups<Shape>(kv_rows),
                                          projection::gemv_groups<Shape>(kv_rows)};
    switch (projection::locate_segment(group, groups)) {
    case 0:
        projection::gemv_segment<Shape>(shared, x, M, kblocks, group, query_rows, query, projection::NoWeight{},
                                        query_out);
        break;
    case 1:
        projection::gemv_segment<Shape>(shared, x, M, kblocks, group, kv_rows, key, projection::NoWeight{}, key_out);
        break;
    case 2:
        projection::gemv_segment<Shape>(shared, x, M, kblocks, group, kv_rows, value, projection::NoWeight{},
                                        value_out);
        break;
    }
}

#define WEIGHTS                                                                                                  \
    KERNEL_W0_AT(SEISMIC_PTR(SEISMIC_BUFFER_QUERY_GATE_WEIGHT)), KERNEL_W1_AT(SEISMIC_PTR(SEISMIC_BUFFER_KEY_WEIGHT)), \
        KERNEL_W2_AT(SEISMIC_PTR(SEISMIC_BUFFER_VALUE_WEIGHT))

extern "C" __global__ void qwen_attention_project_stage(SEISMIC_KERNEL_PARAMS) {
    projection::stage_row<S8>(PROLOGUE, blockIdx.x, (unsigned)SEISMIC_DIM_M, SEISMIC_DIM_D, STAGING, GROUPS);
}

extern "C" __global__ void qwen_attention_project_gemv(SEISMIC_KERNEL_PARAMS) {
    extern __shared__ uint4 dynamic_shared[];
    project_gemv<1>(PROLOGUE, reinterpret_cast<projection::u8 *>(dynamic_shared), STAGING, (unsigned)SEISMIC_DIM_M,
                    SEISMIC_DIM_D, QUERY_ROWS, KV_ROWS, WEIGHTS, QUERY_OUT, KEY_OUT, VALUE_OUT);
}

extern "C" __global__ void qwen_attention_project_gemv16(SEISMIC_KERNEL_PARAMS) {
    project_gemv<2>(PROLOGUE, nullptr, STAGING, (unsigned)SEISMIC_DIM_M, SEISMIC_DIM_D, QUERY_ROWS, KV_ROWS, WEIGHTS,
                    QUERY_OUT, KEY_OUT, VALUE_OUT);
}

extern "C" __global__ void qwen_attention_project_gemm(SEISMIC_KERNEL_PARAMS) {
    extern __shared__ uint4 dynamic_shared[];
    projection::u8 *shared = reinterpret_cast<projection::u8 *>(dynamic_shared);
    const unsigned M = (unsigned)SEISMIC_DIM_M;
    const unsigned long long K = SEISMIC_DIM_D;
    unsigned long long column = blockIdx.x;
    const unsigned long long columns[3] = {projection::gemm_columns(QUERY_ROWS), projection::gemm_columns(KV_ROWS),
                                           projection::gemm_columns(KV_ROWS)};
    switch (projection::locate_segment(column, columns)) {
    case 0:
        projection::gemm_run<GShape, S8>(shared, STAGING, K, GROUPS, M, K, column, QUERY_ROWS,
                                 KERNEL_W0_AT(SEISMIC_PTR(SEISMIC_BUFFER_QUERY_GATE_WEIGHT)), projection::NoWeight{}, QUERY_OUT);
        break;
    case 1:
        projection::gemm_run<GShape, S8>(shared, STAGING, K, GROUPS, M, K, column, KV_ROWS,
                                 KERNEL_W1_AT(SEISMIC_PTR(SEISMIC_BUFFER_KEY_WEIGHT)), projection::NoWeight{}, KEY_OUT);
        break;
    case 2:
        projection::gemm_run<GShape, S8>(shared, STAGING, K, GROUPS, M, K, column, KV_ROWS,
                                 KERNEL_W2_AT(SEISMIC_PTR(SEISMIC_BUFFER_VALUE_WEIGHT)), projection::NoWeight{}, VALUE_OUT);
        break;
    }
}
