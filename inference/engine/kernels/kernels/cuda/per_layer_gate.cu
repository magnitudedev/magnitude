// per_layer_gate: the gate projection of the rounded hidden rows, the forms
// of the `dense` projection (`lib/projection/operator.cuh`) over rows the
// stage launch always forms.
#define KERNEL_W0 SEISMIC_GATE_WEIGHT
#include "lib/projection/operator.cuh"


// The staged rows, read in place by every form.
using Pro = projection::Plain<ELEMENT_OF(SEISMIC_ELEMENT_A), projection::AllRows>;
using Epi = projection::ActivatedMul<ELEMENT_OF(SEISMIC_ELEMENT_A)>;

#define PROJECTION_M SEISMIC_DIM_M
#define PROJECTION_K SEISMIC_DIM_D
#define PROJECTION_N SEISMIC_DIM_P
#define PROJECTION_G 1
#define PROJECTION_WA packets::W0
#define PROJECTION_WB projection::NoWeight
#define PROJECTION_WEIGHT_TYPES projection::op::Types<packets::W0>
#define PROJECTION_INT8 PACKETS_FORM_INT8(KERNEL_W0)
#define PROJECTION_ROWS(g) Pro{PROJECTION_STAGING, SEISMIC_DIM_D, projection::AllRows{}}
#define PROJECTION_PLACE(g) PROJECTION_STAGING
#define PROJECTION_PLACE_STRIDE SEISMIC_DIM_D
#define PROJECTION_STAGING SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_STAGED)
#define PROJECTION_GROUPS SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_GROUPS)
#define PROJECTION_PARTIALS(g, parts) reinterpret_cast<float *>(SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_PARTIALS))
#define PROJECTION_WEIGHTS(g) KERNEL_W0_AT(SEISMIC_PTR(SEISMIC_BUFFER_GATE_WEIGHT)), projection::NoWeight{}
#define PROJECTION_EPILOGUE(g)                                                                            \
    Epi {                                                                                                 \
        SEISMIC_RESULT_0_OUT, SEISMIC_RESULT_0_STRIDE_0,                                  \
            reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_INPUTS)) +                         \
                (long long)SEISMIC_PARAM_LAYER * SEISMIC_INPUTS_STRIDE_1,                                  \
            SEISMIC_INPUTS_STRIDE_0, (int)SEISMIC_PARAM_ACTIVATION                                        \
    }

// The strided F32 hidden source publishes semantic Act before its selected
// matrix representation, formed as the gate weight's operand rows.
#ifdef SEISMIC_FORMING_PER_LAYER_GATE_STAGE
template <unsigned INT8, unsigned OPERAND>
__global__ void per_layer_gate_stage(SEISMIC_KERNEL_PARAMS) {
    using Raw = projection::Strided<element::F32, projection::AllRows>;
    const projection::Operands<projection::OperandRecipe<packets::W0_OPERANDS_NVFP4>::NVFP4, Raw> source{
        Raw{SEISMIC_PTR(SEISMIC_BUFFER_HIDDEN), SEISMIC_HIDDEN_STRIDE_0, SEISMIC_HIDDEN_STRIDE_1, projection::AllRows{}}};
    PROJECTION_STAGE_OF(source);
}
#endif

#ifdef SEISMIC_FORMING_PER_LAYER_GATE_GEMV
template <unsigned INT8, unsigned STAGING, unsigned KSPLIT, unsigned RESIDENCY>
__launch_bounds__(256, PROJECTION_GEMV_BLOCKS(KSPLIT, RESIDENCY)) __global__ void per_layer_gate_gemv(SEISMIC_KERNEL_PARAMS) {
    PROJECTION_GEMV(1, KSPLIT, STAGING, PROJECTION_GEMV_GRID(KSPLIT, RESIDENCY), RESIDENCY);
}
#endif

#ifdef SEISMIC_FORMING_PER_LAYER_GATE_GEMV16
template <unsigned INT8, unsigned STAGING, unsigned KSPLIT>
__global__ void per_layer_gate_gemv16(SEISMIC_KERNEL_PARAMS) {
    PROJECTION_GEMV(2, KSPLIT, STAGING, 0, 1);
}
#endif

#ifdef SEISMIC_FORMING_PER_LAYER_GATE_GEMM_SMALL
template <unsigned INT8, unsigned OPERAND, unsigned SPLIT>
__global__ void per_layer_gate_gemm_small(SEISMIC_KERNEL_PARAMS) {
    PROJECTION_GEMM(projection::SmallGemm, PROJECTION_OPERANDS);
}
#endif

#ifdef SEISMIC_FORMING_PER_LAYER_GATE_GEMM
template <unsigned INT8, unsigned OPERAND, unsigned SPLIT, unsigned ROTATE, unsigned BLOCKS>
__launch_bounds__(256, BLOCKS) __global__ void per_layer_gate_gemm(SEISMIC_KERNEL_PARAMS) {
    PROJECTION_GEMM(projection::LargeGemm<ROTATE>, PROJECTION_OPERANDS);
}
#endif

#ifdef SEISMIC_FORMING_PER_LAYER_GATE_FINALIZE
template <unsigned SPLIT>
__global__ void per_layer_gate_finalize(SEISMIC_KERNEL_PARAMS) {
    PROJECTION_FINALIZE;
}
#endif
