// per_layer_gate: `stage` rounds the F32 hidden rows to A in scratch
// (`stage_s8` also forms the INT8 candidate's q8_1 rows, for the rows whose
// GEMM reads them), then `project_rows`' bands project
// them through the gate rows with the activated-product epilogue
// A(A(act(A(gate))) * inputs[m, layer, p]).
#define KERNEL_W0 SEISMIC_GATE_WEIGHT
#include "lib/projection/projection.cuh"

using Pro = projection::Plain<ELEMENT_OF(SEISMIC_ELEMENT_A), projection::AllRows>;
using Epi = projection::ActivatedMul<ELEMENT_OF(SEISMIC_ELEMENT_A)>;

#define ROUNDED SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_ROUNDED)
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

// Hidden row `m`, rounded to A.
#define ROUND_ROW(m)                                                                                      \
    for (unsigned long long i = threadIdx.x; i < SEISMIC_DIM_D; i += blockDim.x)                          \
        element::put<ELEMENT_OF(SEISMIC_ELEMENT_A)>(                                                      \
            ROUNDED, (m) * SEISMIC_DIM_D + i,                                                             \
            reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_HIDDEN))[(m) * SEISMIC_HIDDEN_STRIDE_0 + \
                                                                                i * SEISMIC_HIDDEN_STRIDE_1])

#ifdef SEISMIC_FORMING_PER_LAYER_GATE_STAGE
template <unsigned INT8>
__global__ void per_layer_gate_stage(SEISMIC_KERNEL_PARAMS) {
    const unsigned long long m = blockIdx.x;
    ROUND_ROW(m);
}
#endif

// The rounded row, and with S8 its q8_1 row.
#ifdef SEISMIC_FORMING_PER_LAYER_GATE_STAGE_S8
template <unsigned INT8>
__global__ void per_layer_gate_stage_s8(SEISMIC_KERNEL_PARAMS) {
    const unsigned long long m = blockIdx.x;
    ROUND_ROW(m);
    constexpr bool S8 = INT8 == 1 && projection::quantizable<packets::W0>;
    if constexpr (S8) {
        __syncthreads();
        projection::stage_row<true>(ROUNDED_ROWS, (unsigned)m, SEISMIC_DIM_D, STAGING, GROUPS);
    }
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
template <unsigned INT8>
__global__ void per_layer_gate_gemm_small(SEISMIC_KERNEL_PARAMS) {
    constexpr bool S8 = INT8 == 1 && projection::quantizable<packets::W0>;
    extern __shared__ uint4 dynamic_shared[];
    const unsigned long long column = projection::gemm_column<projection::SmallGemm>();
    if (column < projection::gemm_columns(SEISMIC_DIM_P))
        projection::gemm_run<projection::SmallGemm, S8>(reinterpret_cast<projection::u8 *>(dynamic_shared),
                                                        S8 ? STAGING : ROUNDED, SEISMIC_DIM_D, GROUPS,
                                                        (unsigned)SEISMIC_DIM_M, SEISMIC_DIM_D, column,
                                                        SEISMIC_DIM_P, GATE, projection::NoWeight{}, EPILOGUE);
}
#endif

#ifdef SEISMIC_FORMING_PER_LAYER_GATE_GEMM
template <unsigned ROTATE>
__global__ void per_layer_gate_gemm(SEISMIC_KERNEL_PARAMS) {
    using Shape = projection::LargeGemm<ROTATE>;
    extern __shared__ uint4 dynamic_shared[];
    const unsigned long long column = projection::gemm_column<Shape>();
    if (column < projection::gemm_columns(SEISMIC_DIM_P))
        projection::gemm_run<Shape, false>(reinterpret_cast<projection::u8 *>(dynamic_shared), ROUNDED, SEISMIC_DIM_D,
                                           GROUPS, (unsigned)SEISMIC_DIM_M, SEISMIC_DIM_D, column, SEISMIC_DIM_P,
                                           GATE, projection::NoWeight{}, EPILOGUE);
}
#endif
