// project_rows: a plain projection of A rows, the forms of the `dense`
// projection (`lib/projection/operator.cuh`).
//
// Group g of a grouped weight (grid z, `dense.seismic`) is the projection of
// the source columns g * K .., through weight rows g * N .., into output
// columns g * N ..: the same launches over its views. The staged forms run
// one group (their declaration), so the stage launch stages group 0's rows.
#define KERNEL_W0 SEISMIC_WEIGHT
#include "lib/projection/operator.cuh"


// The source rows, read in place unless the weight takes them as NVFP4
// operands, which the GEMV forms at M = 1 and the stage launch above.
using SourceElement = ELEMENT_OF(SEISMIC_ELEMENT_A);
using Raw = projection::Plain<SourceElement, projection::AllRows>;
using Pro = projection::Operands<projection::OperandRecipe<packets::W0_OPERANDS_NVFP4>::NVFP4, Raw>;
using Scaling = projection::scaling<(SEISMIC_DIM_WS != 0)>;

// Group `group`'s source columns, weight rows, partials and output columns.
#define GROUP_SOURCE(group) \
    (SEISMIC_PTR(SEISMIC_BUFFER_SOURCE) + (projection::u64)(group) * SEISMIC_DIM_K * SourceElement::bytes)

#define PROJECTION_M SEISMIC_DIM_M
#define PROJECTION_K SEISMIC_DIM_K
#define PROJECTION_N SEISMIC_DIM_N
#define PROJECTION_G SEISMIC_DIM_G
#define PROJECTION_WA packets::W0
#define PROJECTION_WB projection::NoWeight
#define PROJECTION_WEIGHT_TYPES projection::op::Types<packets::W0>
#define PROJECTION_INT8 PACKETS_FORM_INT8(KERNEL_W0)
#define PROJECTION_ROWS(group) Pro{Raw{GROUP_SOURCE(group), SEISMIC_SOURCE_STRIDE_0, projection::AllRows{}}}
#define PROJECTION_PLACE(group) GROUP_SOURCE(group)
#define PROJECTION_PLACE_STRIDE SEISMIC_SOURCE_STRIDE_0
#define PROJECTION_STAGING SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_STAGED)
#define PROJECTION_GROUPS SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_GROUPS)
// Group `group`'s share of the split partials: `parts` parts of M x N each.
#define PROJECTION_PARTIALS(group, parts)                                                                 \
    (reinterpret_cast<float *>(SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_PARTIALS)) +                            \
     (projection::u64)(group) * (parts) * SEISMIC_DIM_M * SEISMIC_DIM_N)
#define PROJECTION_WEIGHTS(group)                                                                         \
    KERNEL_W0_AT(SEISMIC_PTR(SEISMIC_BUFFER_WEIGHT) + (projection::u64)(group) * SEISMIC_DIM_N * KERNEL_W0_ROW_BYTES), \
        projection::NoWeight{}
#define PROJECTION_EPILOGUE(group)                                                                        \
    Scaling::wrap(projection::Store<ELEMENT_OF(SEISMIC_ELEMENT_Y)>{SEISMIC_RESULT_0_OUT,   \
                                                                  SEISMIC_RESULT_0_STRIDE_0,              \
                                                                  (projection::u64)(group) * SEISMIC_DIM_N}, \
                  projection::scale_factor(SEISMIC_PTR(SEISMIC_BUFFER_WEIGHT_SCALE), SEISMIC_DIM_WS, 0, 0), 1.0f)

#ifdef SEISMIC_FORMING_PROJECT_ROWS_STAGE
template <unsigned INT8, unsigned OPERAND>
__global__ void project_rows_stage(SEISMIC_KERNEL_PARAMS) {
    PROJECTION_STAGE;
}
#endif

#ifdef SEISMIC_FORMING_PROJECT_ROWS_GEMV
template <unsigned INT8, unsigned STAGING, unsigned KSPLIT, unsigned RESIDENCY>
__launch_bounds__(256, PROJECTION_GEMV_BLOCKS(KSPLIT, RESIDENCY)) __global__ void project_rows_gemv(SEISMIC_KERNEL_PARAMS) {
    PROJECTION_GEMV(1, KSPLIT, STAGING, PROJECTION_GEMV_GRID(KSPLIT, RESIDENCY), RESIDENCY);
}
#endif

#ifdef SEISMIC_FORMING_PROJECT_ROWS_GEMV16
template <unsigned INT8, unsigned STAGING, unsigned KSPLIT>
__global__ void project_rows_gemv16(SEISMIC_KERNEL_PARAMS) {
    PROJECTION_GEMV(2, KSPLIT, STAGING, 0, 1);
}
#endif

#ifdef SEISMIC_FORMING_PROJECT_ROWS_GEMM_SMALL
template <unsigned INT8, unsigned OPERAND, unsigned SPLIT>
__global__ void project_rows_gemm_small(SEISMIC_KERNEL_PARAMS) {
    PROJECTION_GEMM(projection::SmallGemm, PROJECTION_OPERANDS);
}
#endif

#ifdef SEISMIC_FORMING_PROJECT_ROWS_GEMM
template <unsigned INT8, unsigned OPERAND, unsigned SPLIT, unsigned ROTATE, unsigned BLOCKS>
__launch_bounds__(256, BLOCKS) __global__ void project_rows_gemm(SEISMIC_KERNEL_PARAMS) {
    PROJECTION_GEMM(projection::LargeGemm<ROTATE>, PROJECTION_OPERANDS);
}
#endif

#ifdef SEISMIC_FORMING_PROJECT_ROWS_FINALIZE
template <unsigned SPLIT>
__global__ void project_rows_finalize(SEISMIC_KERNEL_PARAMS) {
    PROJECTION_FINALIZE;
}
#endif
