// per_layer_gate: the shared projection producer publishes semantic Act rows
// or explicitly selected matrix operands. GEMV retains semantic inputs;
// small and large matrix row bands share the same operand eligibility.
#define KERNEL_W0 SEISMIC_GATE_WEIGHT
#include "lib/projection/projection.cuh"

using Pro = projection::Plain<ELEMENT_OF(SEISMIC_ELEMENT_A), projection::AllRows>;
using Epi = projection::ActivatedMul<ELEMENT_OF(SEISMIC_ELEMENT_A)>;

#define ROUNDED SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_STAGED)
#define ROUNDED_ROWS Pro{ROUNDED, SEISMIC_DIM_D, projection::AllRows{}}
#define STAGING SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_STAGED)
#define GROUPS SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_GROUPS)
#define GATE KERNEL_W0_AT(SEISMIC_PTR(SEISMIC_BUFFER_GATE_WEIGHT))
#define EPILOGUE                                                                                          \
    Epi {                                                                                                 \
        SEISMIC_PTR(SEISMIC_RESULT_0_BUFFER), SEISMIC_RESULT_0_STRIDE_0,                                  \
            reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_INPUTS)) +                         \
                (long long)SEISMIC_PARAM_LAYER * SEISMIC_INPUTS_STRIDE_1,                                  \
            SEISMIC_INPUTS_STRIDE_0, (int)SEISMIC_PARAM_ACTIVATION                                        \
    }

// The GEMV over NB column blocks of 8 rows.
template <int NB, int KSPLIT>
__device__ __forceinline__ void gate_gemv(const Pro &rounded, unsigned M, unsigned long long D, unsigned long long P,
                                          const packets::W0 &gate, const Epi &epi) {
    using Shape = projection::GemvShape<8, 1, KSPLIT, NB>;
    __shared__ projection::GemvShared<Shape, Pro> shared;
    const unsigned long long group = Shape::tile_group();
    if (group < projection::gemv_groups<Shape>(P))
        projection::gemv_segment<Shape>(shared, rounded, M, D / 64, group, P, gate, projection::NoWeight{}, epi);
}

// The strided F32 hidden source publishes semantic Act before its selected
// matrix representation. One producer replaces the separate rounding recipes.
#ifdef SEISMIC_FORMING_PER_LAYER_GATE_STAGE
template <unsigned INT8, unsigned OPERAND>
__global__ void per_layer_gate_stage(SEISMIC_KERNEL_PARAMS) {
    constexpr bool S8 = INT8 == 1 && projection::quantizable<packets::W0>;
    using Operands = typename projection::GemmOperands<OPERAND>::type;
    const projection::Strided<element::F32, projection::AllRows> source{
        SEISMIC_PTR(SEISMIC_BUFFER_HIDDEN), SEISMIC_HIDDEN_STRIDE_0, SEISMIC_HIDDEN_STRIDE_1, projection::AllRows{}};
    projection::stage_projection_row<S8, Operands>(source, blockIdx.x, SEISMIC_DIM_M, SEISMIC_DIM_D, STAGING, GROUPS);
}
#endif

#ifdef SEISMIC_FORMING_PER_LAYER_GATE_GEMV
template <unsigned KSPLIT>
__global__ void per_layer_gate_gemv(SEISMIC_KERNEL_PARAMS) {
    gate_gemv<1, KSPLIT>(ROUNDED_ROWS, (unsigned)SEISMIC_DIM_M, SEISMIC_DIM_D, SEISMIC_DIM_P, GATE, EPILOGUE);
}
#endif

#ifdef SEISMIC_FORMING_PER_LAYER_GATE_GEMV16
template <unsigned KSPLIT>
__global__ void per_layer_gate_gemv16(SEISMIC_KERNEL_PARAMS) {
    gate_gemv<2, KSPLIT>(ROUNDED_ROWS, (unsigned)SEISMIC_DIM_M, SEISMIC_DIM_D, SEISMIC_DIM_P, GATE, EPILOGUE);
}
#endif

#ifdef SEISMIC_FORMING_PER_LAYER_GATE_GEMM_SMALL
template <unsigned INT8, unsigned OPERAND>
__global__ void per_layer_gate_gemm_small(SEISMIC_KERNEL_PARAMS) {
    constexpr bool S8 = INT8 == 1 && projection::quantizable<packets::W0>;
    extern __shared__ uint4 dynamic_shared[];
    const unsigned long long column = projection::gemm_column<projection::SmallGemm>();
    if (column < projection::gemm_columns(SEISMIC_DIM_P))
        projection::gemm_run<projection::SmallGemm, S8, typename projection::GemmOperands<OPERAND>::type>(reinterpret_cast<projection::u8 *>(dynamic_shared),
                                                        STAGING, SEISMIC_DIM_D, GROUPS,
                                                        (unsigned)SEISMIC_DIM_M, SEISMIC_DIM_D, column,
                                                        SEISMIC_DIM_P, GATE, projection::NoWeight{}, EPILOGUE);
}
#endif

#ifdef SEISMIC_FORMING_PER_LAYER_GATE_GEMM
template <unsigned INT8, unsigned OPERAND, unsigned ROTATE>
__global__ void per_layer_gate_gemm(SEISMIC_KERNEL_PARAMS) {
    constexpr bool S8 = INT8 == 1 && projection::quantizable<packets::W0>;
    using Shape = projection::LargeGemm<ROTATE>;
    extern __shared__ uint4 dynamic_shared[];
    const unsigned long long column = projection::gemm_column<Shape>();
    if (column < projection::gemm_columns(SEISMIC_DIM_P))
        projection::gemm_run<Shape, S8, typename projection::GemmOperands<OPERAND>::type>(reinterpret_cast<projection::u8 *>(dynamic_shared), ROUNDED, SEISMIC_DIM_D,
                                           GROUPS, (unsigned)SEISMIC_DIM_M, SEISMIC_DIM_D, column, SEISMIC_DIM_P,
                                           GATE, projection::NoWeight{}, EPILOGUE);
}
#endif
