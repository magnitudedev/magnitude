// attention_output: the shared projection producer publishes semantic Act rows
// or explicitly selected matrix operands. GEMV retains semantic inputs;
// small and large matrix row bands share the same operand eligibility.
#define KERNEL_W0 SEISMIC_OUTPUT_WEIGHT
#include "lib/projection/projection.cuh"

SEISMIC_PROGRAMMATIC_DEPENDENCY

// The gated head rows, read in place unless the output weight takes them as
// NVFP4 operands, which the GEMV forms at M = 1 and the stage launch above.
using Raw = projection::Plain<ELEMENT_OF(SEISMIC_ELEMENT_A), projection::AllRows>;
using Pro = projection::Operands<projection::OperandRecipe<packets::W0_OPERANDS_NVFP4>::NVFP4, Raw>;
using Source = projection::GemvSource<Pro>;
// The residual join: the residual add, or (hyper-connection streams) the
// scatter of the projection into each stream, after the weight's
// second-level scale where it has one.
using Output = projection::OutputOf<projection::AllRows, SEISMIC_DIM_SN, (SEISMIC_DIM_IN != 0)>;
using Scaling = projection::scaling<(SEISMIC_DIM_OS != 0)>;
using Epi = Scaling::type<Output::type>;

#define HEADS (SEISMIC_DIM_Q * SEISMIC_DIM_W)
#define GATED_ROWS Pro{Raw{SEISMIC_PTR(SEISMIC_BUFFER_GATED), SEISMIC_GATED_STRIDE_0, projection::AllRows{}}}
// The rows a GEMM reads: staged, or the gated rows in place.
#define GEMM_ROWS(STAGED) ((STAGED) ? STAGING : SEISMIC_PTR(SEISMIC_BUFFER_GATED))
#define GEMM_STRIDE(STAGED) ((STAGED) ? HEADS : SEISMIC_GATED_STRIDE_0)
#define STAGING SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_STAGED)
#define GROUPS SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_GROUPS)
#define OUTPUT KERNEL_W0_AT(SEISMIC_PTR(SEISMIC_BUFFER_OUTPUT_WEIGHT))
#define PARTIALS reinterpret_cast<float *>(SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_PARTIALS))
#define EPILOGUE                                                                                     \
    Scaling::wrap(                                                                                   \
        Output::make(reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_HIDDEN)), SEISMIC_HIDDEN_STRIDE_0, \
                     projection::AllRows{}, reinterpret_cast<float *>(SEISMIC_PTR(SEISMIC_RESULT_0_BUFFER)), \
                     SEISMIC_RESULT_0_STRIDE_0, reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_INJECT)), \
                     SEISMIC_INJECT_STRIDE_0, SEISMIC_DIM_D),                                        \
        projection::scale_factor(SEISMIC_PTR(SEISMIC_BUFFER_OUTPUT_SCALE), SEISMIC_DIM_OS, 0, 0), 1.0f)

// The GEMV over NB column blocks of 8 rows; at M = 1 a block of NVFP4
// operands forms its row (block-collective).
template <int NB, int KSPLIT>
__device__ __forceinline__ void output_gemv(const Pro &heads, projection::u8 *row, const projection::u8 *staged,
                                            unsigned M, unsigned long long K, unsigned long long D,
                                            const packets::W0 &output, const Epi &epi) {
    using Shape = projection::GemvShape<8, 1, KSPLIT, NB>;
    __shared__ projection::GemvShared<Shape, Source::type, projection::gemv_staging<Shape, packets::W0>(HEADS / 64)>
        shared;
    projection::gemv_segment_ready<Shape>(shared, projection::source_after_dependency(heads, row, M, K, staged), M,
                                          K / 64, Shape::tile_group(), D, output, projection::NoWeight{}, epi);
}

#ifdef SEISMIC_FORMING_ATTENTION_OUTPUT_STAGE
template <unsigned INT8, unsigned OPERAND>
__global__ void attention_output_stage(SEISMIC_KERNEL_PARAMS) {
    seismic_dependency_start();
    constexpr bool S8 = INT8 == 1 && projection::quantizable<packets::W0>;
    using Operands = projection::GemmOperandsOf<OPERAND, packets::W0>;
    projection::stage_projection_row<S8, Operands>(GATED_ROWS, blockIdx.x, SEISMIC_DIM_M, HEADS, STAGING, GROUPS);
}
#endif

#ifdef SEISMIC_FORMING_ATTENTION_OUTPUT_GEMV
template <unsigned KSPLIT>
__global__ void attention_output_gemv(SEISMIC_KERNEL_PARAMS) {
    extern __shared__ uint4 dynamic_shared[];
    output_gemv<1, KSPLIT>(GATED_ROWS, reinterpret_cast<projection::u8 *>(dynamic_shared), STAGING,
                           (unsigned)SEISMIC_DIM_M, HEADS, SEISMIC_DIM_D, OUTPUT, EPILOGUE);
}
#endif

#ifdef SEISMIC_FORMING_ATTENTION_OUTPUT_GEMV16
template <unsigned KSPLIT>
__global__ void attention_output_gemv16(SEISMIC_KERNEL_PARAMS) {
    output_gemv<2, KSPLIT>(GATED_ROWS, nullptr, STAGING, (unsigned)SEISMIC_DIM_M, HEADS, SEISMIC_DIM_D, OUTPUT,
                           EPILOGUE);
}
#endif

#ifdef SEISMIC_FORMING_ATTENTION_OUTPUT_GEMM_SMALL
template <unsigned INT8, unsigned OPERAND>
__global__ void attention_output_gemm_small(SEISMIC_KERNEL_PARAMS) {
    using Operands = projection::GemmOperandsOf<OPERAND, packets::W0>;
    seismic_dependency_start();
    constexpr bool S8 = INT8 == 1 && projection::quantizable<packets::W0>;
    extern __shared__ uint4 dynamic_shared[];
    const unsigned long long column = projection::gemm_column<projection::SmallGemm>();
    constexpr bool STAGED = S8 || Operands::MATERIALIZE || Pro::STAGED;
    if (column < projection::gemm_columns(SEISMIC_DIM_D))
        projection::gemm_run_split<projection::SmallGemm, S8, Operands>(
            reinterpret_cast<projection::u8 *>(dynamic_shared), GEMM_ROWS(STAGED), GEMM_STRIDE(STAGED), GROUPS,
            (unsigned)SEISMIC_DIM_M, HEADS, column, SEISMIC_DIM_D, OUTPUT, EPILOGUE, PARTIALS);
}
#endif

#ifdef SEISMIC_FORMING_ATTENTION_OUTPUT_GEMM
template <unsigned INT8, unsigned OPERAND, unsigned ROTATE>
__global__ void attention_output_gemm(SEISMIC_KERNEL_PARAMS) {
    using Operands = projection::GemmOperandsOf<OPERAND, packets::W0>;
    constexpr bool S8 = INT8 == 1 && projection::quantizable<packets::W0>;
    seismic_dependency_start();
    using Shape = projection::LargeGemm<ROTATE>;
    extern __shared__ uint4 dynamic_shared[];
    const unsigned long long column = projection::gemm_column<Shape>();
    constexpr bool STAGED = S8 || Operands::MATERIALIZE || Pro::STAGED;
    if (column < projection::gemm_columns(SEISMIC_DIM_D))
        projection::gemm_run_split<Shape, S8, Operands>(
            reinterpret_cast<projection::u8 *>(dynamic_shared), GEMM_ROWS(STAGED), GEMM_STRIDE(STAGED), GROUPS,
            (unsigned)SEISMIC_DIM_M, HEADS, column, SEISMIC_DIM_D, OUTPUT, EPILOGUE, PARTIALS);
}
#endif

#ifdef SEISMIC_FORMING_ATTENTION_OUTPUT_FINALIZE
extern "C" __global__ void attention_output_finalize(SEISMIC_KERNEL_PARAMS) {
    seismic_dependency_start();
    const Epi epi = EPILOGUE;
    projection::split_finalize<1>(PARTIALS, projection::SPLIT, SEISMIC_DIM_M, SEISMIC_DIM_D,
                                  [&](unsigned m, unsigned long long n, float value, float) { epi(m, n, value, 0.0f); });
}
#endif
