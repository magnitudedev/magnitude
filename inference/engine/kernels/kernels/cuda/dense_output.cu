// dense_output: the down projection of the product rows plus the F32
// residual, the forms of the `dense` projection (`lib/projection/operator.cuh`).
#define KERNEL_W0 SEISMIC_DOWN_WEIGHT
#include "lib/projection/operator.cuh"

SEISMIC_PROGRAMMATIC_DEPENDENCY

// The product rows, read in place unless the down weight takes them as
// NVFP4 operands, which the GEMV forms at M = 1 and the stage launch above.
using Raw = projection::Plain<ELEMENT_OF(SEISMIC_ELEMENT_A), projection::AllRows>;
using Pro = projection::Operands<projection::OperandRecipe<packets::W0_OPERANDS_NVFP4>::NVFP4, Raw>;
using Scaling = projection::scaling<(SEISMIC_DIM_DS != 0)>;

#define PROJECTION_G 1
#define PROJECTION_M SEISMIC_DIM_O
#define PROJECTION_K SEISMIC_DIM_F
#define PROJECTION_N SEISMIC_DIM_H
#define PROJECTION_WA packets::W0
#define PROJECTION_WB projection::NoWeight
#define PROJECTION_WEIGHT_TYPES projection::op::Types<packets::W0>
#define PROJECTION_INT8 PACKETS_FORM_INT8(KERNEL_W0)
#define PROJECTION_ROWS(g) Pro{Raw{SEISMIC_PTR(SEISMIC_BUFFER_PRODUCT), SEISMIC_PRODUCT_STRIDE_0, projection::AllRows{}}}
#define PROJECTION_PLACE(g) SEISMIC_PTR(SEISMIC_BUFFER_PRODUCT)
#define PROJECTION_PLACE_STRIDE SEISMIC_PRODUCT_STRIDE_0
#define PROJECTION_STAGING SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_STAGED)
#define PROJECTION_GROUPS SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_GROUPS)
#define PROJECTION_PARTIALS(g, parts) reinterpret_cast<float *>(SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_PARTIALS))
#define PROJECTION_WEIGHTS(g) KERNEL_W0_AT(SEISMIC_PTR(SEISMIC_BUFFER_DOWN_WEIGHT)), projection::NoWeight{}
#define PROJECTION_EPILOGUE(g)                                                                               \
    Scaling::wrap(                                                                                        \
        projection::Residual<projection::Act, projection::SelectedRows>{                                  \
            reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_RESIDUAL)), SEISMIC_RESIDUAL_STRIDE_0, \
            projection::SelectedRows{reinterpret_cast<const int *>(SEISMIC_PTR(SEISMIC_BUFFER_OUT_ROWS))},    \
            SEISMIC_RESULT_0_OUT, SEISMIC_RESULT_0_STRIDE_0},     \
        projection::scale_factor(SEISMIC_PTR(SEISMIC_BUFFER_DOWN_SCALE), SEISMIC_DIM_DS, 0, 0), 1.0f)

#ifdef SEISMIC_FORMING_DENSE_OUTPUT_STAGE
template <unsigned INT8, unsigned OPERAND>
__global__ void dense_output_stage(SEISMIC_KERNEL_PARAMS) {
    PROJECTION_STAGE;
}
#endif

#ifdef SEISMIC_FORMING_DENSE_OUTPUT_GEMV
template <unsigned STAGING, unsigned KSPLIT, unsigned RESIDENCY>
__launch_bounds__(256, PROJECTION_GEMV_BLOCKS(KSPLIT, RESIDENCY)) __global__ void dense_output_gemv(SEISMIC_KERNEL_PARAMS) {
    PROJECTION_GEMV(1, KSPLIT, STAGING, PROJECTION_GEMV_GRID(KSPLIT, RESIDENCY), RESIDENCY);
}
#endif

#ifdef SEISMIC_FORMING_DENSE_OUTPUT_GEMV16
template <unsigned STAGING, unsigned KSPLIT>
__global__ void dense_output_gemv16(SEISMIC_KERNEL_PARAMS) {
    PROJECTION_GEMV(2, KSPLIT, STAGING, 0, 1);
}
#endif

#ifdef SEISMIC_FORMING_DENSE_OUTPUT_GEMM_SMALL
template <unsigned INT8, unsigned OPERAND, unsigned SPLIT>
__global__ void dense_output_gemm_small(SEISMIC_KERNEL_PARAMS) {
    PROJECTION_GEMM(projection::SmallGemm, PROJECTION_OPERANDS);
}
#endif

#ifdef SEISMIC_FORMING_DENSE_OUTPUT_GEMM
template <unsigned INT8, unsigned OPERAND, unsigned SPLIT, unsigned ROTATE, unsigned BLOCKS>
__launch_bounds__(256, BLOCKS) __global__ void dense_output_gemm(SEISMIC_KERNEL_PARAMS) {
    PROJECTION_GEMM(projection::LargeGemm<ROTATE>, PROJECTION_OPERANDS);
}
#endif

#ifdef SEISMIC_FORMING_DENSE_OUTPUT_FINALIZE
template <unsigned SPLIT>
__global__ void dense_output_finalize(SEISMIC_KERNEL_PARAMS) {
    PROJECTION_FINALIZE;
}
#endif
