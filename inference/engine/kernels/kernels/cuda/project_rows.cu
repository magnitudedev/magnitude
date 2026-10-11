// project_rows: the shared projection producer publishes semantic Act rows
// or explicitly selected matrix operands. GEMV retains semantic inputs;
// small and large matrix row bands share the same operand eligibility.
#define KERNEL_W0 SEISMIC_WEIGHT
#include "lib/projection/projection.cuh"

using Pro = projection::Plain<ELEMENT_OF(SEISMIC_ELEMENT_A), projection::AllRows>;
using Scaling = projection::scaling<(SEISMIC_DIM_WS != 0)>;
using Epi = Scaling::type<projection::Store<ELEMENT_OF(SEISMIC_ELEMENT_Y)>>;

#define SOURCE_ROWS Pro{SEISMIC_PTR(SEISMIC_BUFFER_SOURCE), SEISMIC_SOURCE_STRIDE_0, projection::AllRows{}}
#define STAGING SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_STAGED)
#define GROUPS SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_GROUPS)
#define WEIGHT KERNEL_W0_AT(SEISMIC_PTR(SEISMIC_BUFFER_WEIGHT))
#define PARTIALS reinterpret_cast<float *>(SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_PARTIALS))
#define EPILOGUE                                                                                          \
    Scaling::wrap(projection::Store<ELEMENT_OF(SEISMIC_ELEMENT_Y)>{SEISMIC_PTR(SEISMIC_RESULT_0_BUFFER),   \
                                                                  SEISMIC_RESULT_0_STRIDE_0, 0},          \
                  projection::scale_factor(SEISMIC_PTR(SEISMIC_BUFFER_WEIGHT_SCALE), SEISMIC_DIM_WS, 0, 0), 1.0f)

// The GEMV over NB column blocks of 8 rows.
template <int NB, int KSPLIT>
__device__ __forceinline__ void project_gemv(const Pro &source, unsigned M, unsigned long long K,
                                             unsigned long long N, const packets::W0 &weight, const Epi &epi) {
    using Shape = projection::GemvShape<8, 1, KSPLIT, NB>;
    __shared__ projection::GemvShared<Shape, Pro> shared;
    const unsigned long long group = Shape::tile_group();
    if (group < projection::gemv_groups<Shape>(N))
        projection::gemv_segment<Shape>(shared, source, M, K / 64, group, N, weight, projection::NoWeight{}, epi);
}

#ifdef SEISMIC_FORMING_PROJECT_ROWS_STAGE
template <unsigned INT8, unsigned OPERAND>
__global__ void project_rows_stage(SEISMIC_KERNEL_PARAMS) {
    constexpr bool S8 = INT8 == 1 && projection::quantizable<packets::W0>;
    using Operands = typename projection::GemmOperands<OPERAND>::type;
    projection::stage_projection_row<S8, Operands>(SOURCE_ROWS, blockIdx.x, SEISMIC_DIM_M, SEISMIC_DIM_K, STAGING, GROUPS);
}
#endif

#ifdef SEISMIC_FORMING_PROJECT_ROWS_GEMV
template <unsigned KSPLIT>
__global__ void project_rows_gemv(SEISMIC_KERNEL_PARAMS) {
    project_gemv<1, KSPLIT>(SOURCE_ROWS, (unsigned)SEISMIC_DIM_M, SEISMIC_DIM_K, SEISMIC_DIM_N, WEIGHT, EPILOGUE);
}
#endif

#ifdef SEISMIC_FORMING_PROJECT_ROWS_GEMV16
template <unsigned KSPLIT>
__global__ void project_rows_gemv16(SEISMIC_KERNEL_PARAMS) {
    project_gemv<2, KSPLIT>(SOURCE_ROWS, (unsigned)SEISMIC_DIM_M, SEISMIC_DIM_K, SEISMIC_DIM_N, WEIGHT, EPILOGUE);
}
#endif

#ifdef SEISMIC_FORMING_PROJECT_ROWS_GEMM_SMALL
template <unsigned INT8, unsigned OPERAND>
__global__ void project_rows_gemm_small(SEISMIC_KERNEL_PARAMS) {
    using Operands = typename projection::GemmOperands<OPERAND>::type;
    constexpr bool S8 = INT8 == 1 && projection::quantizable<packets::W0>;
    extern __shared__ uint4 dynamic_shared[];
    const unsigned long long column = projection::gemm_column<projection::SmallGemm>();
    if (column < projection::gemm_columns(SEISMIC_DIM_N))
        projection::gemm_run_split<projection::SmallGemm, S8, Operands>(
            reinterpret_cast<projection::u8 *>(dynamic_shared), (S8 || Operands::MATERIALIZE) ? STAGING : SEISMIC_PTR(SEISMIC_BUFFER_SOURCE),
            (S8 || Operands::MATERIALIZE) ? SEISMIC_DIM_K : SEISMIC_SOURCE_STRIDE_0, GROUPS, (unsigned)SEISMIC_DIM_M, SEISMIC_DIM_K, column,
            SEISMIC_DIM_N, WEIGHT, EPILOGUE, PARTIALS);
}
#endif

#ifdef SEISMIC_FORMING_PROJECT_ROWS_GEMM
template <unsigned INT8, unsigned OPERAND, unsigned ROTATE>
__global__ void project_rows_gemm(SEISMIC_KERNEL_PARAMS) {
    using Operands = typename projection::GemmOperands<OPERAND>::type;
    constexpr bool S8 = INT8 == 1 && projection::quantizable<packets::W0>;
    using Shape = projection::LargeGemm<ROTATE>;
    extern __shared__ uint4 dynamic_shared[];
    const unsigned long long column = projection::gemm_column<Shape>();
    if (column < projection::gemm_columns(SEISMIC_DIM_N))
        projection::gemm_run_split<Shape, S8, Operands>(
            reinterpret_cast<projection::u8 *>(dynamic_shared), (S8 || Operands::MATERIALIZE) ? STAGING : SEISMIC_PTR(SEISMIC_BUFFER_SOURCE),
            (S8 || Operands::MATERIALIZE) ? SEISMIC_DIM_K : SEISMIC_SOURCE_STRIDE_0, GROUPS, (unsigned)SEISMIC_DIM_M, SEISMIC_DIM_K, column, SEISMIC_DIM_N, WEIGHT,
            EPILOGUE, PARTIALS);
}
#endif

#ifdef SEISMIC_FORMING_PROJECT_ROWS_FINALIZE
extern "C" __global__ void project_rows_finalize(SEISMIC_KERNEL_PARAMS) {
    const Epi epi = EPILOGUE;
    projection::split_finalize<1>(PARTIALS, projection::SPLIT, SEISMIC_DIM_M, SEISMIC_DIM_N,
                                  [&](unsigned m, unsigned long long n, float value, float) { epi(m, n, value, 0.0f); });
}
#endif
