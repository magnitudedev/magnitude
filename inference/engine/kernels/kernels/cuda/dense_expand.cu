// dense_expand: the gate and up projections of the normalized rows combined
// by the GLU epilogue, the forms of the `dense` projection
// (`lib/projection/operator.cuh`).
#define KERNEL_W0 SEISMIC_GATE_WEIGHT
#define KERNEL_W1 SEISMIC_UP_WEIGHT
// The expansion's limit as its F32 bits LB (0: none), which the Glu epilogue
// clamps at.
#if SEISMIC_STATIC_LB != 0
#define PROJECTION_GLU_LIMIT SEISMIC_STATIC_LB
#endif
#include "lib/projection/operator.cuh"


using Raw = projection::Rms<ELEMENT_OF(SEISMIC_NORM), projection::SelectedRows>;
using Pro = projection::Operands<
    projection::OperandRecipe<packets::W0_OPERANDS_NVFP4, packets::W1_OPERANDS_NVFP4>::NVFP4, Raw>;
using Scaling = projection::scaling<(SEISMIC_DIM_GS != 0 || SEISMIC_DIM_US != 0)>;

#define PROJECTION_G 1
#define PROJECTION_M SEISMIC_DIM_O
#define PROJECTION_K SEISMIC_DIM_H
#define PROJECTION_N SEISMIC_DIM_F
#define PROJECTION_WA packets::W0
#define PROJECTION_WB packets::W1
#define PROJECTION_WEIGHT_TYPES projection::op::Types<packets::W0, packets::W1>
#define PROJECTION_INT8 PACKETS_FORM_INT8(KERNEL_W0), PACKETS_FORM_INT8(KERNEL_W1)
#define PROJECTION_ROWS(g)                                                                                       \
    Pro {                                                                                                     \
        Raw {                                                                                                 \
            reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_RESIDUAL)), SEISMIC_RESIDUAL_STRIDE_0,  \
                SEISMIC_PTR(SEISMIC_BUFFER_NORM), __uint_as_float((unsigned)SEISMIC_PARAM_EPS), SEISMIC_DIM_H, \
                projection::SelectedRows{reinterpret_cast<const int *>(SEISMIC_PTR(SEISMIC_BUFFER_OUT_ROWS))}  \
        }                                                                                                     \
    }
#define PROJECTION_PLACE(g) SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_STAGED)
#define PROJECTION_PLACE_STRIDE SEISMIC_DIM_H
#define PROJECTION_STAGING SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_STAGED)
#define PROJECTION_GROUPS SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_GROUPS)
#define PROJECTION_PARTIALS(g, parts) nullptr
#define PROJECTION_WEIGHTS(g) KERNEL_W0_AT(SEISMIC_PTR(SEISMIC_BUFFER_GATE_WEIGHT)), KERNEL_W1_AT(SEISMIC_PTR(SEISMIC_BUFFER_UP_WEIGHT))
#define PROJECTION_EPILOGUE(g)                                                                               \
    Scaling::wrap(projection::Glu<ELEMENT_OF(SEISMIC_ELEMENT_A)>{SEISMIC_RESULT_0_OUT,     \
                                                                 SEISMIC_RESULT_0_STRIDE_0,               \
                                                                 (int)SEISMIC_PARAM_ACTIVATION},          \
                  projection::scale_factor(SEISMIC_PTR(SEISMIC_BUFFER_GATE_SCALE), SEISMIC_DIM_GS, 0, 0), \
                  projection::scale_factor(SEISMIC_PTR(SEISMIC_BUFFER_UP_SCALE), SEISMIC_DIM_US, 0, 0))

#ifdef SEISMIC_FORMING_DENSE_EXPAND_STAGE
template <unsigned INT8, unsigned OPERAND>
__global__ void dense_expand_stage(SEISMIC_KERNEL_PARAMS) {
    PROJECTION_STAGE;
}
#endif

#ifdef SEISMIC_FORMING_DENSE_EXPAND_GEMV
template <unsigned STAGING, unsigned KSPLIT, unsigned RESIDENCY>
__launch_bounds__(256, PROJECTION_GEMV_BLOCKS(KSPLIT, RESIDENCY)) __global__ void dense_expand_gemv(SEISMIC_KERNEL_PARAMS) {
    extern __shared__ uint4 dynamic_shared[];
    PROJECTION_GEMV(1, KSPLIT, STAGING, PROJECTION_GEMV_GRID(KSPLIT, RESIDENCY), RESIDENCY,
                    reinterpret_cast<projection::u8 *>(dynamic_shared));
}
#endif

#ifdef SEISMIC_FORMING_DENSE_EXPAND_GEMV16
template <unsigned STAGING, unsigned KSPLIT>
__global__ void dense_expand_gemv16(SEISMIC_KERNEL_PARAMS) {
    PROJECTION_GEMV(2, KSPLIT, STAGING, 0, 1, nullptr);
}
#endif

#ifdef SEISMIC_FORMING_DENSE_EXPAND_GEMM_SMALL
template <unsigned INT8, unsigned OPERAND>
__global__ void dense_expand_gemm_small(SEISMIC_KERNEL_PARAMS) {
    PROJECTION_GEMM(projection::SmallGemm, PROJECTION_OPERANDS);
}
#endif

#ifdef SEISMIC_FORMING_DENSE_EXPAND_GEMM
template <unsigned INT8, unsigned OPERAND, unsigned ROTATE>
__global__ void dense_expand_gemm(SEISMIC_KERNEL_PARAMS) {
    PROJECTION_GEMM(projection::LargeGemm<ROTATE>, PROJECTION_OPERANDS);
}
#endif
