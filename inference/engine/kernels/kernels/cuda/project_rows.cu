// project_rows: the shared projection producer publishes semantic Act rows
// or explicitly selected matrix operands. GEMV retains semantic inputs;
// small and large matrix row bands share the same operand eligibility.
//
// Group g of a grouped weight (grid z, `cuda_dense.seismic`) is the
// projection of the source columns g * K .., through weight rows g * N ..,
// into output columns g * N ..: the same launches over its views. The staged
// forms run one group (their declaration), so the stage launch stages group
// 0's rows.
#define KERNEL_W0 SEISMIC_WEIGHT
#include "lib/projection/projection.cuh"

// The source rows, read in place unless the weight takes them as NVFP4
// operands, which the GEMV forms at M = 1 and the stage launch above.
using SourceElement = ELEMENT_OF(SEISMIC_ELEMENT_A);
using Raw = projection::Plain<SourceElement, projection::AllRows>;
using Pro = projection::Operands<projection::OperandRecipe<packets::W0_OPERANDS_NVFP4>::NVFP4, Raw>;
using Source = projection::GemvSource<Pro>;
using Scaling = projection::scaling<(SEISMIC_DIM_WS != 0)>;
using Epi = Scaling::type<projection::Store<ELEMENT_OF(SEISMIC_ELEMENT_Y)>>;

// Group `group`'s source columns, weight rows and output columns.
#define GROUP_SOURCE(group)                                                                               \
    (SEISMIC_PTR(SEISMIC_BUFFER_SOURCE) +                                                                 \
     (projection::u64)(group) * SEISMIC_DIM_K * SourceElement::bytes)
#define SOURCE_ROWS(group) Pro{Raw{GROUP_SOURCE(group), SEISMIC_SOURCE_STRIDE_0, projection::AllRows{}}}
// The rows a GEMM reads: staged, or the group's source rows in place.
#define GEMM_ROWS(STAGED, group) ((STAGED) ? STAGING : GROUP_SOURCE(group))
#define GEMM_STRIDE(STAGED) ((STAGED) ? SEISMIC_DIM_K : SEISMIC_SOURCE_STRIDE_0)
#define STAGING SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_STAGED)
#define GROUPS SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_GROUPS)
#define WEIGHT(group)                                                                                     \
    KERNEL_W0_AT(SEISMIC_PTR(SEISMIC_BUFFER_WEIGHT) +                                                     \
                 (projection::u64)(group) * SEISMIC_DIM_N * KERNEL_W0_ROW_BYTES)
// Group `group`'s share of the split partials: SPLIT parts of M x N each.
#define PARTIALS(group)                                                                                   \
    (reinterpret_cast<float *>(SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_PARTIALS)) +                            \
     (projection::u64)(group) * projection::SPLIT * SEISMIC_DIM_M * SEISMIC_DIM_N)
#define EPILOGUE(group)                                                                                   \
    Scaling::wrap(projection::Store<ELEMENT_OF(SEISMIC_ELEMENT_Y)>{SEISMIC_PTR(SEISMIC_RESULT_0_BUFFER),   \
                                                                  SEISMIC_RESULT_0_STRIDE_0,              \
                                                                  (projection::u64)(group) * SEISMIC_DIM_N}, \
                  projection::scale_factor(SEISMIC_PTR(SEISMIC_BUFFER_WEIGHT_SCALE), SEISMIC_DIM_WS, 0, 0), 1.0f)

// The GEMV over NB column blocks of 8 rows; at M = 1 a block of NVFP4
// operands forms its row (block-collective, before any group's check).
template <int NB, int KSPLIT>
__device__ __forceinline__ void project_gemv(const Pro &source, projection::u8 *row, const projection::u8 *staged,
                                             unsigned M, unsigned long long K, unsigned long long N,
                                             const packets::W0 &weight, const Epi &epi) {
    using Shape = projection::GemvShape<8, 1, KSPLIT, NB>;
    __shared__ projection::GemvShared<Shape, Source::type> shared;
    const Source::type x = Source::make(source, row, M, K, staged);
    const unsigned long long group = Shape::tile_group();
    projection::gemv_segment<Shape>(shared, x, M, K / 64, group, N, weight, projection::NoWeight{}, epi);
}

// A GEMM block of group z / splits, K share z % splits (grid z = G * splits),
// in a kernel's body (it reads the kernel's parameters).
#define PROJECT_ROWS_GEMM(Shape, S8, Operands, STAGED, shared)                                            \
    do {                                                                                                  \
        const unsigned long long column = projection::gemm_column<Shape>();                               \
        const unsigned long long splits = gridDim.z / SEISMIC_DIM_G;                                      \
        const unsigned long long group = blockIdx.z / splits;                                             \
        if (column < projection::gemm_columns(SEISMIC_DIM_N))                                             \
            projection::gemm_run_split<Shape, S8, Operands>(                                              \
                shared, GEMM_ROWS(STAGED, group), GEMM_STRIDE(STAGED), GROUPS, (unsigned)SEISMIC_DIM_M,   \
                SEISMIC_DIM_K, column, SEISMIC_DIM_N, WEIGHT(group), EPILOGUE(group), PARTIALS(group),    \
                blockIdx.z % splits, splits);                                                             \
    } while (0)

#ifdef SEISMIC_FORMING_PROJECT_ROWS_STAGE
template <unsigned INT8, unsigned OPERAND>
__global__ void project_rows_stage(SEISMIC_KERNEL_PARAMS) {
    constexpr bool S8 = projection::int8_path<INT8, PACKETS_FORM_INT8(KERNEL_W0)>();
    using Operands = projection::GemmOperandsOf<OPERAND, packets::W0>;
    projection::stage_projection_row<S8, Operands>(SOURCE_ROWS(0), blockIdx.x, SEISMIC_DIM_M, SEISMIC_DIM_K, STAGING,
                                                   GROUPS);
}
#endif

#ifdef SEISMIC_FORMING_PROJECT_ROWS_GEMV
template <unsigned KSPLIT>
__global__ void project_rows_gemv(SEISMIC_KERNEL_PARAMS) {
    extern __shared__ uint4 dynamic_shared[];
    project_gemv<1, KSPLIT>(SOURCE_ROWS(blockIdx.z), reinterpret_cast<projection::u8 *>(dynamic_shared), STAGING,
                            (unsigned)SEISMIC_DIM_M, SEISMIC_DIM_K, SEISMIC_DIM_N, WEIGHT(blockIdx.z),
                            EPILOGUE(blockIdx.z));
}
#endif

#ifdef SEISMIC_FORMING_PROJECT_ROWS_GEMV16
template <unsigned KSPLIT>
__global__ void project_rows_gemv16(SEISMIC_KERNEL_PARAMS) {
    project_gemv<2, KSPLIT>(SOURCE_ROWS(blockIdx.z), nullptr, STAGING, (unsigned)SEISMIC_DIM_M, SEISMIC_DIM_K,
                            SEISMIC_DIM_N, WEIGHT(blockIdx.z), EPILOGUE(blockIdx.z));
}
#endif

#ifdef SEISMIC_FORMING_PROJECT_ROWS_GEMM_SMALL
template <unsigned INT8, unsigned OPERAND>
__global__ void project_rows_gemm_small(SEISMIC_KERNEL_PARAMS) {
    using Operands = projection::GemmOperandsOf<OPERAND, packets::W0>;
    constexpr bool S8 = projection::int8_path<INT8, PACKETS_FORM_INT8(KERNEL_W0)>();
    extern __shared__ uint4 dynamic_shared[];
    constexpr bool STAGED = S8 || Operands::MATERIALIZE || Pro::STAGED;
    PROJECT_ROWS_GEMM(projection::SmallGemm, S8, Operands, STAGED, reinterpret_cast<projection::u8 *>(dynamic_shared));
}
#endif

#ifdef SEISMIC_FORMING_PROJECT_ROWS_GEMM
template <unsigned INT8, unsigned OPERAND, unsigned ROTATE>
__global__ void project_rows_gemm(SEISMIC_KERNEL_PARAMS) {
    using Operands = projection::GemmOperandsOf<OPERAND, packets::W0>;
    constexpr bool S8 = projection::int8_path<INT8, PACKETS_FORM_INT8(KERNEL_W0)>();
    extern __shared__ uint4 dynamic_shared[];
    constexpr bool STAGED = S8 || Operands::MATERIALIZE || Pro::STAGED;
    using Shape = projection::LargeGemm<ROTATE>;
    PROJECT_ROWS_GEMM(Shape, S8, Operands, STAGED, reinterpret_cast<projection::u8 *>(dynamic_shared));
}
#endif

#ifdef SEISMIC_FORMING_PROJECT_ROWS_FINALIZE
extern "C" __global__ void project_rows_finalize(SEISMIC_KERNEL_PARAMS) {
    const Epi epi = EPILOGUE(blockIdx.z);
    projection::split_finalize<1>(PARTIALS(blockIdx.z), projection::SPLIT, SEISMIC_DIM_M, SEISMIC_DIM_N,
                                  [&](unsigned m, unsigned long long n, float value, float) { epi(m, n, value, 0.0f); });
}
#endif
