// stream_mix_up: the low-rank rows (read in place) through `up`, each output
// column the gate of its stream's column, publishing the per-stream
// partials (`Mix`): the forms of the `dense` projection
// (`lib/projection/operator.cuh`).
#define KERNEL_A SEISMIC_ELEMENT_A
#define KERNEL_W0 SEISMIC_UP_WEIGHT
#include "lib/projection/operator.cuh"

SEISMIC_PROGRAMMATIC_DEPENDENCY

// The low-rank rows, read in place (the up weight takes no NVFP4 operands).
using Pro = projection::Plain<ELEMENT_OF(SEISMIC_ELEMENT_A), projection::AllRows>;
using Epi = projection::Mix<ELEMENT_OF(SEISMIC_NORM), SEISMIC_DIM_H>;

#define PROJECTION_M SEISMIC_DIM_M
#define PROJECTION_K SEISMIC_DIM_LR
#define PROJECTION_N (SEISMIC_DIM_S * SEISMIC_DIM_H)
#define PROJECTION_G 1
#define PROJECTION_WA packets::W0
#define PROJECTION_WB projection::NoWeight
#define PROJECTION_WEIGHT_TYPES projection::op::Types<packets::W0>
#define PROJECTION_INT8 PACKETS_FORM_INT8(KERNEL_W0)
#define PROJECTION_ROWS(g) Pro{SEISMIC_PTR(SEISMIC_BUFFER_LOW), SEISMIC_LOW_STRIDE_0, projection::AllRows{}}
#define PROJECTION_PLACE(g) SEISMIC_PTR(SEISMIC_BUFFER_LOW)
#define PROJECTION_PLACE_STRIDE SEISMIC_LOW_STRIDE_0
#define PROJECTION_STAGING SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_STAGED)
#define PROJECTION_GROUPS SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_GROUPS)
#define PROJECTION_PARTIALS(g, parts) reinterpret_cast<float *>(SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_PARTIALS))
#define PROJECTION_WEIGHTS(g) KERNEL_W0_AT(SEISMIC_PTR(SEISMIC_BUFFER_UP_WEIGHT)), projection::NoWeight{}
#define PROJECTION_EPILOGUE(g)                                                                           \
    Epi {                                                                                                \
        SEISMIC_RESULT_0_OUT, SEISMIC_RESULT_0_STRIDE_0,      \
            reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_HIDDEN)), SEISMIC_HIDDEN_STRIDE_0, \
            reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_INVERSE)), SEISMIC_INVERSE_STRIDE_0, \
            SEISMIC_PTR(SEISMIC_BUFFER_NORM), SEISMIC_DIM_H, (float)SEISMIC_DIM_S                        \
    }

#ifdef SEISMIC_FORMING_STREAM_MIX_UP_STAGE
template <unsigned INT8, unsigned OPERAND>
__global__ void stream_mix_up_stage(SEISMIC_KERNEL_PARAMS) {
    PROJECTION_STAGE;
}
#endif

#ifdef SEISMIC_FORMING_STREAM_MIX_UP_GEMV
template <unsigned INT8, unsigned STAGING, unsigned KSPLIT, unsigned RESIDENCY>
__launch_bounds__(256, PROJECTION_GEMV_BLOCKS(KSPLIT, RESIDENCY)) __global__ void stream_mix_up_gemv(SEISMIC_KERNEL_PARAMS) {
    PROJECTION_GEMV(1, KSPLIT, STAGING, PROJECTION_GEMV_GRID(KSPLIT, RESIDENCY), RESIDENCY);
}
#endif

#ifdef SEISMIC_FORMING_STREAM_MIX_UP_GEMV16
template <unsigned INT8, unsigned STAGING, unsigned KSPLIT>
__global__ void stream_mix_up_gemv16(SEISMIC_KERNEL_PARAMS) {
    PROJECTION_GEMV(2, KSPLIT, STAGING, 0, 1);
}
#endif

#ifdef SEISMIC_FORMING_STREAM_MIX_UP_GEMM_SMALL
template <unsigned INT8, unsigned OPERAND, unsigned SPLIT>
__global__ void stream_mix_up_gemm_small(SEISMIC_KERNEL_PARAMS) {
    PROJECTION_GEMM(projection::SmallGemm, PROJECTION_OPERANDS);
}
#endif

#ifdef SEISMIC_FORMING_STREAM_MIX_UP_GEMM
template <unsigned INT8, unsigned OPERAND, unsigned SPLIT, unsigned ROTATE, unsigned BLOCKS>
__launch_bounds__(256, BLOCKS) __global__ void stream_mix_up_gemm(SEISMIC_KERNEL_PARAMS) {
    PROJECTION_GEMM(projection::LargeGemm<ROTATE>, PROJECTION_OPERANDS);
}
#endif

#ifdef SEISMIC_FORMING_STREAM_MIX_UP_FINALIZE
template <unsigned SPLIT>
__global__ void stream_mix_up_finalize(SEISMIC_KERNEL_PARAMS) {
    PROJECTION_FINALIZE;
}
#endif
