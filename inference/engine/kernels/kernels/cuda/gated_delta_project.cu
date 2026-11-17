// gated_delta_project: the qkv, z, alpha and beta projections of the
// normalized input rows, the forms of the `dense` projection
// (`lib/projection/operator.cuh`), segmented.
#define KERNEL_W0 SEISMIC_QKV_WEIGHT
#define KERNEL_W1 SEISMIC_GATE_WEIGHT
#define KERNEL_W2 SEISMIC_ALPHA_WEIGHT
#define KERNEL_W3 SEISMIC_BETA_WEIGHT
#include "lib/projection/operator.cuh"

SEISMIC_PROGRAMMATIC_DEPENDENCY

using Raw = projection::Input<ELEMENT_OF(SEISMIC_INPUT_NORM), projection::AllRows, SEISMIC_DIM_NM, SEISMIC_DIM_PT>;
using Pro = projection::Operands<
    projection::OperandRecipe<packets::W0_OPERANDS_NVFP4, packets::W1_OPERANDS_NVFP4, packets::W2_OPERANDS_NVFP4,
                              packets::W3_OPERANDS_NVFP4>::NVFP4,
    Raw>;
using Store = projection::Store<ELEMENT_OF(SEISMIC_ELEMENT_A)>;

#define QKV_ROWS ((2 * SEISMIC_DIM_NK + SEISMIC_DIM_NV) * SEISMIC_DIM_W)
#define Z_ROWS (SEISMIC_DIM_NV * SEISMIC_DIM_W)
#define GATE_ROWS SEISMIC_DIM_NV
#define OUT(offset) Store{SEISMIC_RESULT_0_OUT, SEISMIC_RESULT_0_STRIDE_0, (offset)}

// The four segments: rows, weights and results in segment order.
#define SEGMENT_ROWS QKV_ROWS, Z_ROWS, GATE_ROWS, GATE_ROWS
using Segments = projection::op::SegmentSet<0, projection::op::Types<packets::W0, packets::W1, packets::W2, packets::W3>,
                                            projection::op::Types<Store, Store, Store, Store>,
                                            projection::op::empty_mask(SEGMENT_ROWS)>;

#define PROJECTION_M SEISMIC_DIM_M
#define PROJECTION_K SEISMIC_DIM_H
#define PROJECTION_ZERO 0
#define PROJECTION_INT8                                                                                       \
    PACKETS_FORM_INT8(KERNEL_W0), PACKETS_FORM_INT8(KERNEL_W1), PACKETS_FORM_INT8(KERNEL_W2),                  \
        PACKETS_FORM_INT8(KERNEL_W3)
#define PROJECTION_WEIGHT_TYPES Segments::Weights
#define PROJECTION_ROWS(g)                                                                                \
    Pro {                                                                                                 \
        Raw {                                                                                             \
            reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_HIDDEN)), SEISMIC_HIDDEN_STRIDE_0,  \
                SEISMIC_PTR(SEISMIC_BUFFER_INPUT_NORM), __uint_as_float((unsigned)SEISMIC_PARAM_EPSILON),  \
                SEISMIC_DIM_H, projection::AllRows{}                                                      \
        }                                                                                                 \
    }
#define PROJECTION_STAGING SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_STAGED)
#define PROJECTION_GROUPS SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_GROUPS)
#define PROJECTION_SEGMENTS                                                                               \
    Segments {                                                                                            \
        {SEGMENT_ROWS},                                                                                   \
            {KERNEL_W0_AT(SEISMIC_PTR(SEISMIC_BUFFER_QKV_WEIGHT)),                                         \
             KERNEL_W1_AT(SEISMIC_PTR(SEISMIC_BUFFER_GATE_WEIGHT)),                                        \
             KERNEL_W2_AT(SEISMIC_PTR(SEISMIC_BUFFER_ALPHA_WEIGHT)),                                       \
             KERNEL_W3_AT(SEISMIC_PTR(SEISMIC_BUFFER_BETA_WEIGHT))},                                       \
        {OUT(0), OUT(QKV_ROWS), OUT(QKV_ROWS + Z_ROWS), OUT(QKV_ROWS + Z_ROWS + GATE_ROWS)}               \
    }

#ifdef SEISMIC_FORMING_GATED_DELTA_PROJECT_STAGE
template <unsigned INT8, unsigned OPERAND>
__global__ void gated_delta_project_stage(SEISMIC_KERNEL_PARAMS) {
    PROJECTION_STAGE;
}
#endif

#ifdef SEISMIC_FORMING_GATED_DELTA_PROJECT_GEMV
template <unsigned INT8, unsigned STAGING, unsigned KSPLIT>
__global__ void gated_delta_project_gemv(SEISMIC_KERNEL_PARAMS) {
    PROJECTION_SEGMENTED_GEMV(1, KSPLIT, STAGING);
}
#endif

#ifdef SEISMIC_FORMING_GATED_DELTA_PROJECT_GEMV16
template <unsigned INT8, unsigned STAGING, unsigned KSPLIT>
__global__ void gated_delta_project_gemv16(SEISMIC_KERNEL_PARAMS) {
    PROJECTION_SEGMENTED_GEMV(2, KSPLIT, STAGING);
}
#endif

#ifdef SEISMIC_FORMING_GATED_DELTA_PROJECT_GEMM_SMALL
template <unsigned INT8, unsigned OPERAND>
__global__ void gated_delta_project_gemm_small(SEISMIC_KERNEL_PARAMS) {
    PROJECTION_SEGMENTED_GEMM(projection::SmallGemm, PROJECTION_OPERANDS);
}
#endif

#ifdef SEISMIC_FORMING_GATED_DELTA_PROJECT_GEMM
template <unsigned INT8, unsigned OPERAND, unsigned ROTATE, unsigned BLOCKS>
__launch_bounds__(256, BLOCKS) __global__ void gated_delta_project_gemm(SEISMIC_KERNEL_PARAMS) {
    PROJECTION_SEGMENTED_GEMM(projection::LargeGemm<ROTATE>, PROJECTION_OPERANDS);
}
#endif
