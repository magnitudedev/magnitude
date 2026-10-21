// dense_output: the shared projection producer publishes semantic Act rows
// or explicitly selected matrix operands. GEMV retains semantic inputs;
// small and large matrix row bands share the same operand eligibility.
#define KERNEL_W0 SEISMIC_DOWN_WEIGHT
#include "lib/projection/projection.cuh"

SEISMIC_PROGRAMMATIC_DEPENDENCY

using Pro = projection::Plain<ELEMENT_OF(SEISMIC_ELEMENT_A), projection::AllRows>;
using Scaling = projection::scaling<(SEISMIC_DIM_DS != 0)>;
using Epi = Scaling::type<projection::Residual<projection::SelectedRows>>;

#define PRODUCT Pro{SEISMIC_PTR(SEISMIC_BUFFER_PRODUCT), SEISMIC_PRODUCT_STRIDE_0, projection::AllRows{}}
#define STAGING SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_STAGED)
#define GROUPS SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_GROUPS)
#define DOWN KERNEL_W0_AT(SEISMIC_PTR(SEISMIC_BUFFER_DOWN_WEIGHT))
#define PARTIALS reinterpret_cast<float *>(SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_PARTIALS))
#define EPILOGUE                                                                                          \
    Scaling::wrap(                                                                                        \
        projection::Residual<projection::SelectedRows>{                                                   \
            reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_RESIDUAL)), SEISMIC_RESIDUAL_STRIDE_0, \
            projection::SelectedRows{reinterpret_cast<const int *>(SEISMIC_PTR(SEISMIC_BUFFER_OUT_ROWS))},    \
            reinterpret_cast<float *>(SEISMIC_PTR(SEISMIC_RESULT_0_BUFFER)), SEISMIC_RESULT_0_STRIDE_0},     \
        projection::scale_factor(SEISMIC_PTR(SEISMIC_BUFFER_DOWN_SCALE), SEISMIC_DIM_DS, 0, 0), 1.0f)

template <int NB, int KSPLIT> using OutputShape = projection::GemvShape<8, 1, KSPLIT, NB>;

// The GEMV over NB column blocks of 8 rows, a launch of GRID blocks (0: not
// stated, `projection::GemvResidency`); it waits for the previous launch once
// its warps' first weight loads are in flight.
template <int NB, int KSPLIT, unsigned long long GRID>
__device__ __forceinline__ void output_gemv(const Pro &product, unsigned M, unsigned long long H, unsigned long long F,
                                            const packets::W0 &down, const Epi &epi) {
    using Shape = OutputShape<NB, KSPLIT>;
    __shared__ projection::GemvShared<Shape, Pro> shared;
    const unsigned long long group = Shape::tile_group();
    if (group < projection::gemv_groups<Shape>(H))
        projection::gemv_segment_ready<Shape, GRID>(shared, projection::after_dependency(product), M, F / 64, group, H,
                                                    down, projection::NoWeight{}, epi);
    else
        seismic_dependency_start();
}

// The 1..8-row GEMV states its grid, so its blocks share their SMs as one
// wave of them needs, and its launch bounds admit that residency.
#define OUTPUT_GRID(KSPLIT) projection::gemv_grid<OutputShape<1, KSPLIT>>(SEISMIC_DIM_H)
#define OUTPUT_MIN_BLOCKS(KSPLIT) \
    projection::gemv_min_blocks<OutputShape<1, KSPLIT>, packets::W0, projection::NoWeight, OUTPUT_GRID(KSPLIT)>()

#ifdef SEISMIC_FORMING_DENSE_OUTPUT_STAGE
template <unsigned INT8, unsigned OPERAND>
__global__ void dense_output_stage(SEISMIC_KERNEL_PARAMS) {
    seismic_dependency_start();
    constexpr bool S8 = INT8 == 1 && projection::quantizable<packets::W0>;
    using Operands = typename projection::GemmOperands<OPERAND>::type;
    projection::stage_projection_row<S8, Operands>(PRODUCT, blockIdx.x, SEISMIC_DIM_O, SEISMIC_DIM_F, STAGING, GROUPS);
}
#endif

#ifdef SEISMIC_FORMING_DENSE_OUTPUT_GEMV
template <unsigned KSPLIT>
__launch_bounds__(256, OUTPUT_MIN_BLOCKS(KSPLIT)) __global__ void dense_output_gemv(SEISMIC_KERNEL_PARAMS) {
    output_gemv<1, KSPLIT, OUTPUT_GRID(KSPLIT)>(PRODUCT, (unsigned)SEISMIC_DIM_O, SEISMIC_DIM_H, SEISMIC_DIM_F, DOWN,
                                                EPILOGUE);
}
#endif

#ifdef SEISMIC_FORMING_DENSE_OUTPUT_GEMV16
template <unsigned KSPLIT>
__global__ void dense_output_gemv16(SEISMIC_KERNEL_PARAMS) {
    output_gemv<2, KSPLIT, 0>(PRODUCT, (unsigned)SEISMIC_DIM_O, SEISMIC_DIM_H, SEISMIC_DIM_F, DOWN, EPILOGUE);
}
#endif

#ifdef SEISMIC_FORMING_DENSE_OUTPUT_GEMM_SMALL
template <unsigned INT8, unsigned OPERAND>
__global__ void dense_output_gemm_small(SEISMIC_KERNEL_PARAMS) {
    using Operands = typename projection::GemmOperands<OPERAND>::type;
    constexpr bool S8 = INT8 == 1 && projection::quantizable<packets::W0>;
    extern __shared__ uint4 dynamic_shared[];
    seismic_dependency_start();
    const unsigned long long column = projection::gemm_column<projection::SmallGemm>();
    if (column < projection::gemm_columns(SEISMIC_DIM_H))
        projection::gemm_run_split<projection::SmallGemm, S8, Operands>(
            reinterpret_cast<projection::u8 *>(dynamic_shared), (S8 || Operands::MATERIALIZE) ? STAGING : SEISMIC_PTR(SEISMIC_BUFFER_PRODUCT),
            (S8 || Operands::MATERIALIZE) ? SEISMIC_DIM_F : SEISMIC_PRODUCT_STRIDE_0, GROUPS, (unsigned)SEISMIC_DIM_O, SEISMIC_DIM_F, column,
            SEISMIC_DIM_H, DOWN, EPILOGUE, PARTIALS);
}
#endif

#ifdef SEISMIC_FORMING_DENSE_OUTPUT_GEMM
template <unsigned INT8, unsigned OPERAND, unsigned ROTATE>
__global__ void dense_output_gemm(SEISMIC_KERNEL_PARAMS) {
    using Operands = typename projection::GemmOperands<OPERAND>::type;
    constexpr bool S8 = INT8 == 1 && projection::quantizable<packets::W0>;
    using Shape = projection::LargeGemm<ROTATE>;
    extern __shared__ uint4 dynamic_shared[];
    seismic_dependency_start();
    const unsigned long long column = projection::gemm_column<Shape>();
    if (column < projection::gemm_columns(SEISMIC_DIM_H))
        projection::gemm_run_split<Shape, S8, Operands>(
            reinterpret_cast<projection::u8 *>(dynamic_shared), (S8 || Operands::MATERIALIZE) ? STAGING : SEISMIC_PTR(SEISMIC_BUFFER_PRODUCT),
            (S8 || Operands::MATERIALIZE) ? SEISMIC_DIM_F : SEISMIC_PRODUCT_STRIDE_0, GROUPS, (unsigned)SEISMIC_DIM_O, SEISMIC_DIM_F, column, SEISMIC_DIM_H, DOWN,
            EPILOGUE, PARTIALS);
}
#endif

#ifdef SEISMIC_FORMING_DENSE_OUTPUT_FINALIZE
extern "C" __global__ void dense_output_finalize(SEISMIC_KERNEL_PARAMS) {
    seismic_dependency_start();
    const Epi epi = EPILOGUE;
    projection::split_finalize<1>(PARTIALS, projection::SPLIT, SEISMIC_DIM_O, SEISMIC_DIM_H,
                                  [&](unsigned m, unsigned long long n, float value, float) { epi(m, n, value, 0.0f); });
}
#endif
