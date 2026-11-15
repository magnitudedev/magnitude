// attention_output: the output projection of the gated attention rows plus
// the residual join, the forms of the `dense` projection
// (`lib/projection/operator.cuh`).
#define KERNEL_W0 SEISMIC_OUTPUT_WEIGHT
#include "lib/projection/operator.cuh"

SEISMIC_PROGRAMMATIC_DEPENDENCY

// The gated head rows, read in place unless the output weight takes them as
// NVFP4 operands, which the GEMV forms at M = 1 and the stage launch above.
using Raw = projection::Plain<ELEMENT_OF(SEISMIC_ELEMENT_A), projection::AllRows>;
using Pro = projection::Operands<projection::OperandRecipe<packets::W0_OPERANDS_NVFP4>::NVFP4, Raw>;
// The residual join: the residual add, or (hyper-connection streams) the
// scatter of the projection into each stream (mixed by the comb when the
// inject row carries one), after the weight's second-level scale where it
// has one.
using Output = projection::OutputOf<projection::Act, projection::AllRows, SEISMIC_DIM_SN, (SEISMIC_DIM_IN != 0),
                                    (SEISMIC_DIM_IN > SEISMIC_DIM_SN)>;
using Scaling = projection::scaling<(SEISMIC_DIM_OS != 0)>;

#define PROJECTION_G 1
#define PROJECTION_M SEISMIC_DIM_M
#define PROJECTION_K (SEISMIC_DIM_Q * SEISMIC_DIM_W)
#define PROJECTION_N SEISMIC_DIM_D
#define PROJECTION_WA packets::W0
#define PROJECTION_WB projection::NoWeight
#define PROJECTION_WEIGHT_TYPES projection::op::Types<packets::W0>
#define PROJECTION_INT8 PACKETS_FORM_INT8(KERNEL_W0)
#define PROJECTION_ROWS(g) Pro{Raw{SEISMIC_PTR(SEISMIC_BUFFER_GATED), SEISMIC_GATED_STRIDE_0, projection::AllRows{}}}
#define PROJECTION_PLACE(g) SEISMIC_PTR(SEISMIC_BUFFER_GATED)
#define PROJECTION_PLACE_STRIDE SEISMIC_GATED_STRIDE_0
#define PROJECTION_STAGING SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_STAGED)
#define PROJECTION_GROUPS SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_GROUPS)
#define PROJECTION_PARTIALS(g, parts) reinterpret_cast<float *>(SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_PARTIALS))
#define PROJECTION_WEIGHTS(g) KERNEL_W0_AT(SEISMIC_PTR(SEISMIC_BUFFER_OUTPUT_WEIGHT)), projection::NoWeight{}
#define PROJECTION_EPILOGUE(g)                                                                                      \
    Scaling::wrap(                                                                                               \
        Output::make(reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_HIDDEN)), SEISMIC_HIDDEN_STRIDE_0, \
                     projection::AllRows{}, SEISMIC_RESULT_0_OUT,     \
                     SEISMIC_RESULT_0_STRIDE_0, reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_INJECT)), \
                     SEISMIC_INJECT_STRIDE_0, SEISMIC_DIM_D),                                                    \
        projection::scale_factor(SEISMIC_PTR(SEISMIC_BUFFER_OUTPUT_SCALE), SEISMIC_DIM_OS, 0, 0), 1.0f)

#ifdef SEISMIC_FORMING_ATTENTION_OUTPUT_STAGE
template <unsigned INT8, unsigned OPERAND>
__global__ void attention_output_stage(SEISMIC_KERNEL_PARAMS) {
    PROJECTION_STAGE;
}
#endif

#ifdef SEISMIC_FORMING_ATTENTION_OUTPUT_GEMV
template <unsigned STAGING, unsigned KSPLIT, unsigned RESIDENCY>
__launch_bounds__(256, PROJECTION_GEMV_BLOCKS(KSPLIT, RESIDENCY)) __global__ void attention_output_gemv(SEISMIC_KERNEL_PARAMS) {
    extern __shared__ uint4 dynamic_shared[];
    PROJECTION_GEMV(1, KSPLIT, STAGING, PROJECTION_GEMV_GRID(KSPLIT, RESIDENCY), RESIDENCY,
                    reinterpret_cast<projection::u8 *>(dynamic_shared));
}
#endif

#ifdef SEISMIC_FORMING_ATTENTION_OUTPUT_GEMV16
template <unsigned STAGING, unsigned KSPLIT>
__global__ void attention_output_gemv16(SEISMIC_KERNEL_PARAMS) {
    PROJECTION_GEMV(2, KSPLIT, STAGING, 0, 1, nullptr);
}
#endif

#ifdef SEISMIC_FORMING_ATTENTION_OUTPUT_GEMM_SMALL
template <unsigned INT8, unsigned OPERAND, unsigned SPLIT>
__global__ void attention_output_gemm_small(SEISMIC_KERNEL_PARAMS) {
    PROJECTION_GEMM(projection::SmallGemm, PROJECTION_OPERANDS);
}
#endif

#ifdef SEISMIC_FORMING_ATTENTION_OUTPUT_GEMM
template <unsigned INT8, unsigned OPERAND, unsigned SPLIT, unsigned ROTATE>
__global__ void attention_output_gemm(SEISMIC_KERNEL_PARAMS) {
    PROJECTION_GEMM(projection::LargeGemm<ROTATE>, PROJECTION_OPERANDS);
}
#endif

#ifdef SEISMIC_FORMING_ATTENTION_OUTPUT_FINALIZE
template <unsigned SPLIT>
__global__ void attention_output_finalize(SEISMIC_KERNEL_PARAMS) {
    PROJECTION_FINALIZE;
}
#endif
