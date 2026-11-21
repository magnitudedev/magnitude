// short_conv_project: RMS prologue over every residual row, then the forms of
// the `dense` projection (`lib/projection/operator.cuh`) over two segments
// along x: the u segment pairs the B and X streams into u = B * X (F32,
// columns 0..CH), the C segment stores C (F32, columns CH..2CH).
#define KERNEL_A SEISMIC_ELEMENT_A
#define KERNEL_W0 SEISMIC_B_WEIGHT
#define KERNEL_W1 SEISMIC_X_WEIGHT
#define KERNEL_W2 SEISMIC_C_WEIGHT
#include "lib/projection/operator.cuh"


using Raw = projection::Rms<ELEMENT_OF(SEISMIC_NORM), projection::AllRows>;
using Pro = projection::Operands<projection::OperandRecipe<packets::W0_OPERANDS_NVFP4, packets::W1_OPERANDS_NVFP4,
                                                           packets::W2_OPERANDS_NVFP4>::NVFP4,
                                 Raw>;
using Product = projection::Mul;
using Gate = projection::Store<element::F32>;

// The two segments: rows, weights and epilogues (formed in the kernel body,
// where the ABI macros are defined).
#define SEGMENT_ROWS SEISMIC_DIM_CH, SEISMIC_DIM_CH
using Segments = projection::op::SegmentSet<0, projection::op::Types<packets::W0, packets::W1, packets::W2>,
                                            projection::op::Types<Product, Gate>,
                                            projection::op::empty_mask(SEGMENT_ROWS)>;

#define PROJECTION_M SEISMIC_DIM_M
#define PROJECTION_K SEISMIC_DIM_H
#define PROJECTION_ZERO 0
#define PROJECTION_INT8 PACKETS_FORM_INT8(KERNEL_W0), PACKETS_FORM_INT8(KERNEL_W1), PACKETS_FORM_INT8(KERNEL_W2)
#define PROJECTION_WEIGHT_TYPES Segments::Weights
#define PROJECTION_ROWS(g)                                                                                    \
    Pro {                                                                                                     \
        Raw {                                                                                                 \
            reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_RESIDUAL)), SEISMIC_RESIDUAL_STRIDE_0,  \
                SEISMIC_PTR(SEISMIC_BUFFER_NORM), __uint_as_float((unsigned)SEISMIC_PARAM_EPS), SEISMIC_DIM_H, \
                projection::AllRows{}                                                                         \
        }                                                                                                     \
    }
#define PROJECTION_STAGING SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_STAGED)
#define PROJECTION_GROUPS SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_GROUPS)
#define PROJECTION_SEGMENTS                                                                                   \
    Segments {                                                                                                \
        {SEGMENT_ROWS},                                                                                       \
            {KERNEL_W0_AT(SEISMIC_PTR(SEISMIC_BUFFER_B_WEIGHT)), KERNEL_W1_AT(SEISMIC_PTR(SEISMIC_BUFFER_X_WEIGHT)), \
             KERNEL_W2_AT(SEISMIC_PTR(SEISMIC_BUFFER_C_WEIGHT))},                                              \
            {Product{SEISMIC_RESULT_0_OUT, SEISMIC_RESULT_0_STRIDE_0}, \
             Gate{SEISMIC_RESULT_0_OUT, SEISMIC_RESULT_0_STRIDE_0, SEISMIC_DIM_CH}}            \
    }

#ifdef SEISMIC_FORMING_SHORT_CONV_PROJECT_STAGE
template <unsigned INT8, unsigned OPERAND>
__global__ void short_conv_project_stage(SEISMIC_KERNEL_PARAMS) {
    PROJECTION_STAGE;
}
#endif

#ifdef SEISMIC_FORMING_SHORT_CONV_PROJECT_GEMV
template <unsigned INT8, unsigned STAGING, unsigned KSPLIT>
__global__ void short_conv_project_gemv(SEISMIC_KERNEL_PARAMS) {
    PROJECTION_SEGMENTED_GEMV(1, KSPLIT, STAGING);
}
#endif

#ifdef SEISMIC_FORMING_SHORT_CONV_PROJECT_GEMV16
template <unsigned INT8, unsigned STAGING, unsigned KSPLIT>
__global__ void short_conv_project_gemv16(SEISMIC_KERNEL_PARAMS) {
    PROJECTION_SEGMENTED_GEMV(2, KSPLIT, STAGING);
}
#endif

#ifdef SEISMIC_FORMING_SHORT_CONV_PROJECT_GEMM_SMALL
template <unsigned INT8, unsigned OPERAND, unsigned GEMM_SPLIT>
__global__ void short_conv_project_gemm_small(SEISMIC_KERNEL_PARAMS) {
    PROJECTION_SEGMENTED_GEMM(projection::SmallGemm, PROJECTION_OPERANDS);
}
#endif

#ifdef SEISMIC_FORMING_SHORT_CONV_PROJECT_GEMM
template <unsigned INT8, unsigned OPERAND, unsigned GEMM_SPLIT, unsigned ROTATE, unsigned BLOCKS>
__launch_bounds__(256, BLOCKS) __global__ void short_conv_project_gemm(SEISMIC_KERNEL_PARAMS) {
    PROJECTION_SEGMENTED_GEMM(projection::LargeGemm<ROTATE>, PROJECTION_OPERANDS);
}
#endif
