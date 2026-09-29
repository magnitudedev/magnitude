// attention_project: RMS prologue over the F32 hidden rows, then one
// segmented projection query | gate | key | value, each segment with its own
// representation and result (a segment of zero rows has no blocks). GEMV for
// M <= 16 (`gemv` to 8 rows, `gemv16` beyond; at M = 1 the block forms the A
// row in shared memory, else `stage` forms the A rows first), GEMM otherwise:
// `gemm_small` to 64 rows (A rows, or the INT8 candidate's q8_1 rows staged by
// `stage_s8`), `gemm` beyond (A rows).
#define KERNEL_W0 SEISMIC_QUERY_WEIGHT
#define KERNEL_W1 SEISMIC_GATE_WEIGHT
#define KERNEL_W2 SEISMIC_KEY_WEIGHT
#define KERNEL_W3 SEISMIC_VALUE_WEIGHT
#include "lib/projection/projection.cuh"

using Pro = projection::Rms<ELEMENT_OF(SEISMIC_INPUT_NORM), projection::AllRows>;
using Source = projection::GemvSource<Pro>;
using Out = projection::Store<ELEMENT_OF(SEISMIC_ELEMENT_A)>;

#define PROLOGUE                                                                                          \
    Pro {                                                                                                 \
        reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_HIDDEN)), SEISMIC_HIDDEN_STRIDE_0,      \
            SEISMIC_PTR(SEISMIC_BUFFER_INPUT_NORM), __uint_as_float((unsigned)SEISMIC_PARAM_EPSILON),      \
            SEISMIC_DIM_D, projection::AllRows{}                                                                  \
    }
#define STAGING SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_STAGED)
#define GROUPS SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_GROUPS)

// The four segments: rows, weights and results in segment order.
struct Segments {
    unsigned long long rows[4];
    packets::W0 query;
    packets::W1 gate;
    packets::W2 key;
    packets::W3 value;
    Out out[4];
};

#define SEGMENTS                                                                                          \
    Segments {                                                                                            \
        {SEISMIC_DIM_Q, SEISMIC_DIM_GR, SEISMIC_DIM_K, SEISMIC_DIM_V},                                     \
            KERNEL_W0_AT(SEISMIC_PTR(SEISMIC_BUFFER_QUERY_WEIGHT)),                                        \
            KERNEL_W1_AT(SEISMIC_PTR(SEISMIC_BUFFER_GATE_WEIGHT)),                                         \
            KERNEL_W2_AT(SEISMIC_PTR(SEISMIC_BUFFER_KEY_WEIGHT)),                                          \
            KERNEL_W3_AT(SEISMIC_PTR(SEISMIC_BUFFER_VALUE_WEIGHT)),                                        \
        {                                                                                                 \
            Out{SEISMIC_PTR(SEISMIC_RESULT_0_BUFFER), SEISMIC_RESULT_0_STRIDE_0, 0},                       \
                Out{SEISMIC_PTR(SEISMIC_RESULT_1_BUFFER), SEISMIC_RESULT_1_STRIDE_0, 0},                   \
                Out{SEISMIC_PTR(SEISMIC_RESULT_2_BUFFER), SEISMIC_RESULT_2_STRIDE_0, 0},                   \
                Out{SEISMIC_PTR(SEISMIC_RESULT_3_BUFFER), SEISMIC_RESULT_3_STRIDE_0, 0},                   \
        }                                                                                                 \
    }

// The segmented GEMV over NB column blocks of 8 rows: query | gate | key |
// value.
template <int NB, int KSPLIT>
__device__ __forceinline__ void project_gemv(const Pro &pro, projection::u8 *row, const projection::u8 *staged,
                                             unsigned M, unsigned long long D, const Segments &s) {
    using Shape = projection::GemvShape<8, 1, KSPLIT, NB>;
    __shared__ projection::GemvShared<Shape, Source::type> shared;
    const Source::type x = Source::make(pro, row, M, D, staged);
    const unsigned long long kblocks = D / 64;
    unsigned long long group = Shape::tile_group();
    const unsigned long long groups[4] = {
        projection::gemv_groups<Shape>(s.rows[0]), projection::gemv_groups<Shape>(s.rows[1]),
        projection::gemv_groups<Shape>(s.rows[2]), projection::gemv_groups<Shape>(s.rows[3])};
    switch (projection::locate_segment(group, groups)) {
    case 0:
        projection::gemv_segment<Shape>(shared, x, M, kblocks, group, s.rows[0], s.query, projection::NoWeight{},
                                        s.out[0]);
        break;
    case 1:
        projection::gemv_segment<Shape>(shared, x, M, kblocks, group, s.rows[1], s.gate, projection::NoWeight{},
                                        s.out[1]);
        break;
    case 2:
        projection::gemv_segment<Shape>(shared, x, M, kblocks, group, s.rows[2], s.key, projection::NoWeight{},
                                        s.out[2]);
        break;
    case 3:
        projection::gemv_segment<Shape>(shared, x, M, kblocks, group, s.rows[3], s.value, projection::NoWeight{},
                                        s.out[3]);
        break;
    }
}

// The segmented GEMM of one row band over the staged rows (A rows, or q8_1
// rows with Q): block column blockIdx.x of query | gate | key | value.
template <class Shape, bool Q>
__device__ __forceinline__ void project_gemm(const projection::u8 *staged, const void *groups, unsigned M,
                                             unsigned long long D, const Segments &s) {
    extern __shared__ uint4 dynamic_shared[];
    projection::u8 *shared = reinterpret_cast<projection::u8 *>(dynamic_shared);
    unsigned long long column = blockIdx.x;
    const unsigned long long columns[4] = {
        projection::gemm_columns(s.rows[0]), projection::gemm_columns(s.rows[1]),
        projection::gemm_columns(s.rows[2]), projection::gemm_columns(s.rows[3])};
    switch (projection::locate_segment(column, columns)) {
    case 0:
        projection::gemm_run<Shape, Q>(shared, staged, D, groups, M, D, column, s.rows[0], s.query,
                                       projection::NoWeight{}, s.out[0]);
        break;
    case 1:
        projection::gemm_run<Shape, Q>(shared, staged, D, groups, M, D, column, s.rows[1], s.gate,
                                       projection::NoWeight{}, s.out[1]);
        break;
    case 2:
        projection::gemm_run<Shape, Q>(shared, staged, D, groups, M, D, column, s.rows[2], s.key,
                                       projection::NoWeight{}, s.out[2]);
        break;
    case 3:
        projection::gemm_run<Shape, Q>(shared, staged, D, groups, M, D, column, s.rows[3], s.value,
                                       projection::NoWeight{}, s.out[3]);
        break;
    }
}

// The INT8 path needs every non-empty segment packed (an empty segment's
// binding is irrelevant).
#define QUANTIZABLE                                                                                       \
    ((SEISMIC_DIM_Q == 0 || projection::quantizable<packets::W0>) &&                                      \
     (SEISMIC_DIM_GR == 0 || projection::quantizable<packets::W1>) &&                                     \
     (SEISMIC_DIM_K == 0 || projection::quantizable<packets::W2>) &&                                      \
     (SEISMIC_DIM_V == 0 || projection::quantizable<packets::W3>))

#ifdef SEISMIC_FORMING_ATTENTION_PROJECT_STAGE
template <unsigned INT8>
__global__ void attention_project_stage(SEISMIC_KERNEL_PARAMS) {
    projection::stage_row<false>(PROLOGUE, blockIdx.x, SEISMIC_DIM_D, STAGING, GROUPS);
}
#endif

#ifdef SEISMIC_FORMING_ATTENTION_PROJECT_STAGE_S8
template <unsigned INT8>
__global__ void attention_project_stage_s8(SEISMIC_KERNEL_PARAMS) {
    constexpr bool S8 = INT8 == 1 && QUANTIZABLE;
    projection::stage_row<S8>(PROLOGUE, blockIdx.x, SEISMIC_DIM_D, STAGING, GROUPS);
}
#endif

#ifdef SEISMIC_FORMING_ATTENTION_PROJECT_GEMV
template <unsigned KSPLIT>
__global__ void attention_project_gemv(SEISMIC_KERNEL_PARAMS) {
    extern __shared__ uint4 dynamic_shared[];
    project_gemv<1, KSPLIT>(PROLOGUE, reinterpret_cast<projection::u8 *>(dynamic_shared), STAGING,
                            (unsigned)SEISMIC_DIM_M, SEISMIC_DIM_D, SEGMENTS);
}
#endif

#ifdef SEISMIC_FORMING_ATTENTION_PROJECT_GEMV16
template <unsigned KSPLIT>
__global__ void attention_project_gemv16(SEISMIC_KERNEL_PARAMS) {
    project_gemv<2, KSPLIT>(PROLOGUE, nullptr, STAGING, (unsigned)SEISMIC_DIM_M, SEISMIC_DIM_D, SEGMENTS);
}
#endif

#ifdef SEISMIC_FORMING_ATTENTION_PROJECT_GEMM_SMALL
template <unsigned INT8>
__global__ void attention_project_gemm_small(SEISMIC_KERNEL_PARAMS) {
    constexpr bool S8 = INT8 == 1 && QUANTIZABLE;
    project_gemm<projection::SmallGemm, S8>(STAGING, GROUPS, (unsigned)SEISMIC_DIM_M, SEISMIC_DIM_D, SEGMENTS);
}
#endif

#ifdef SEISMIC_FORMING_ATTENTION_PROJECT_GEMM
extern "C" __global__ void attention_project_gemm(SEISMIC_KERNEL_PARAMS) {
    project_gemm<projection::LargeGemm, false>(STAGING, GROUPS, (unsigned)SEISMIC_DIM_M, SEISMIC_DIM_D, SEGMENTS);
}
#endif
