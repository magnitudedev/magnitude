// dense_output: the shared projection producer publishes semantic Act rows
// or explicitly selected matrix operands. GEMV retains semantic inputs;
// small and large matrix row bands share the same operand eligibility.
#define KERNEL_W0 SEISMIC_DOWN_WEIGHT
#include "lib/projection/projection.cuh"

SEISMIC_PROGRAMMATIC_DEPENDENCY

// The product rows, read in place unless the down weight takes them as
// NVFP4 operands, which the GEMV forms at M = 1 and the stage launch above.
using Raw = projection::Plain<ELEMENT_OF(SEISMIC_ELEMENT_A), projection::AllRows>;
using Pro = projection::Operands<projection::OperandRecipe<packets::W0_OPERANDS_NVFP4>::NVFP4, Raw>;
using Source = projection::GemvSource<Pro>;
using Scaling = projection::scaling<(SEISMIC_DIM_DS != 0)>;
using Epi = Scaling::type<projection::Residual<projection::SelectedRows>>;

#define PRODUCT Pro{Raw{SEISMIC_PTR(SEISMIC_BUFFER_PRODUCT), SEISMIC_PRODUCT_STRIDE_0, projection::AllRows{}}}
// The rows a GEMM reads: staged, or the product rows in place.
#define GEMM_ROWS(STAGED) ((STAGED) ? STAGING : SEISMIC_PTR(SEISMIC_BUFFER_PRODUCT))
#define GEMM_STRIDE(STAGED) ((STAGED) ? SEISMIC_DIM_F : SEISMIC_PRODUCT_STRIDE_0)
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
// stated, `projection::GemvResidency`). Every warp runs it, a group past the
// rows included: once the warps' first weight loads are in flight it waits
// for the previous launch, and at M = 1 a block of NVFP4 operands forms its
// row (block-collective).
template <int NB, int KSPLIT, unsigned long long GRID>
__device__ __forceinline__ void output_gemv(const Pro &product, projection::u8 *row, const projection::u8 *staged,
                                            unsigned M, unsigned long long H, unsigned long long F,
                                            const packets::W0 &down, const Epi &epi) {
    using Shape = OutputShape<NB, KSPLIT>;
    __shared__ projection::GemvShared<Shape, Source::type> shared;
    projection::gemv_segment_ready<Shape, GRID>(shared, projection::source_after_dependency(product, row, M, F, staged),
                                                M, F / 64, Shape::tile_group(), H, down, projection::NoWeight{}, epi);
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
    using Operands = projection::GemmOperandsOf<OPERAND, packets::W0>;
    projection::stage_projection_row<S8, Operands>(PRODUCT, blockIdx.x, SEISMIC_DIM_O, SEISMIC_DIM_F, STAGING, GROUPS);
}
#endif

#ifdef SEISMIC_FORMING_DENSE_OUTPUT_GEMV
template <unsigned KSPLIT>
__launch_bounds__(256, OUTPUT_MIN_BLOCKS(KSPLIT)) __global__ void dense_output_gemv(SEISMIC_KERNEL_PARAMS) {
    extern __shared__ uint4 dynamic_shared[];
    output_gemv<1, KSPLIT, OUTPUT_GRID(KSPLIT)>(PRODUCT, reinterpret_cast<projection::u8 *>(dynamic_shared), STAGING,
                                                (unsigned)SEISMIC_DIM_O, SEISMIC_DIM_H, SEISMIC_DIM_F, DOWN, EPILOGUE);
}
#endif

#ifdef SEISMIC_FORMING_DENSE_OUTPUT_GEMV16
template <unsigned KSPLIT>
__global__ void dense_output_gemv16(SEISMIC_KERNEL_PARAMS) {
    output_gemv<2, KSPLIT, 0>(PRODUCT, nullptr, STAGING, (unsigned)SEISMIC_DIM_O, SEISMIC_DIM_H, SEISMIC_DIM_F, DOWN,
                              EPILOGUE);
}
#endif

#ifdef SEISMIC_FORMING_DENSE_OUTPUT_GEMM_SMALL
template <unsigned INT8, unsigned OPERAND>
__global__ void dense_output_gemm_small(SEISMIC_KERNEL_PARAMS) {
    using Operands = projection::GemmOperandsOf<OPERAND, packets::W0>;
    constexpr bool S8 = INT8 == 1 && projection::quantizable<packets::W0>;
    extern __shared__ uint4 dynamic_shared[];
    seismic_dependency_start();
    const unsigned long long column = projection::gemm_column<projection::SmallGemm>();
    constexpr bool STAGED = S8 || Operands::MATERIALIZE || Pro::STAGED;
    if (column < projection::gemm_columns(SEISMIC_DIM_H))
        projection::gemm_run_split<projection::SmallGemm, S8, Operands>(
            reinterpret_cast<projection::u8 *>(dynamic_shared), GEMM_ROWS(STAGED), GEMM_STRIDE(STAGED), GROUPS,
            (unsigned)SEISMIC_DIM_O, SEISMIC_DIM_F, column, SEISMIC_DIM_H, DOWN, EPILOGUE, PARTIALS);
}
#endif

#ifdef SEISMIC_FORMING_DENSE_OUTPUT_GEMM
template <unsigned INT8, unsigned OPERAND, unsigned ROTATE>
__global__ void dense_output_gemm(SEISMIC_KERNEL_PARAMS) {
    using Operands = projection::GemmOperandsOf<OPERAND, packets::W0>;
    constexpr bool S8 = INT8 == 1 && projection::quantizable<packets::W0>;
    using Shape = projection::LargeGemm<ROTATE>;
    extern __shared__ uint4 dynamic_shared[];
    seismic_dependency_start();
    const unsigned long long column = projection::gemm_column<Shape>();
    constexpr bool STAGED = S8 || Operands::MATERIALIZE || Pro::STAGED;
    if (column < projection::gemm_columns(SEISMIC_DIM_H))
        projection::gemm_run_split<Shape, S8, Operands>(
            reinterpret_cast<projection::u8 *>(dynamic_shared), GEMM_ROWS(STAGED), GEMM_STRIDE(STAGED), GROUPS,
            (unsigned)SEISMIC_DIM_O, SEISMIC_DIM_F, column, SEISMIC_DIM_H, DOWN, EPILOGUE, PARTIALS);
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
