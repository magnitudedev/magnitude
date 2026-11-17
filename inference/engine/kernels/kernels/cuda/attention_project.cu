// attention_project: the query, gate, key and value projections of the
// normalized input rows, the forms of the `dense` projection
// (`lib/projection/operator.cuh`), segmented. A call in mode 1 zeroes the
// query and gate segments.
#define KERNEL_W0 SEISMIC_QUERY_WEIGHT
#define KERNEL_W1 SEISMIC_GATE_WEIGHT
#define KERNEL_W2 SEISMIC_KEY_WEIGHT
#define KERNEL_W3 SEISMIC_VALUE_WEIGHT
#include "lib/projection/operator.cuh"

SEISMIC_PROGRAMMATIC_DEPENDENCY

using Raw = projection::Input<ELEMENT_OF(SEISMIC_INPUT_NORM), projection::AllRows, SEISMIC_DIM_NM, SEISMIC_DIM_PT>;
using Pro = projection::Operands<
    projection::OperandRecipe<packets::W0_OPERANDS_NVFP4, packets::W1_OPERANDS_NVFP4, packets::W2_OPERANDS_NVFP4,
                              packets::W3_OPERANDS_NVFP4>::NVFP4,
    Raw>;
// A segment's result, after its weight's second-level scale where any
// segment has a scale port (an absent port's factor is 1), published in Y.
using Store = projection::Store<ELEMENT_OF(SEISMIC_ELEMENT_Y)>;
using Scaling = projection::scaling<(SEISMIC_DIM_QS != 0 || SEISMIC_DIM_GS != 0 || SEISMIC_DIM_KS != 0 ||
                                     SEISMIC_DIM_VS != 0)>;
using Out = Scaling::type<Store>;

// The four segments: rows, weights and results in segment order.
#define SEGMENT_ROWS SEISMIC_DIM_Q, SEISMIC_DIM_GR, SEISMIC_DIM_K, SEISMIC_DIM_V
using Segments = projection::op::SegmentSet<1, projection::op::Types<packets::W0, packets::W1, packets::W2, packets::W3>,
                                            projection::op::Types<Out, Out, Out, Out>,
                                            projection::op::empty_mask(SEGMENT_ROWS)>;

#define SEGMENT_OUT(RESULT, STRIDE, SCALE, EXTENT)                                                        \
    Scaling::wrap(Store{RESULT, STRIDE, 0}, projection::scale_factor(SEISMIC_PTR(SCALE), EXTENT, 0, 0), \
                  1.0f)

#define PROJECTION_M SEISMIC_DIM_M
#define PROJECTION_K SEISMIC_DIM_D
#define PROJECTION_ZERO SEISMIC_PARAM_PROJECT_MODE
// An empty segment's binding is irrelevant to the INT8 path.
#define PROJECTION_INT8                                                                                      \
    SEISMIC_DIM_Q == 0 || PACKETS_FORM_INT8(KERNEL_W0), SEISMIC_DIM_GR == 0 || PACKETS_FORM_INT8(KERNEL_W1), \
        SEISMIC_DIM_K == 0 || PACKETS_FORM_INT8(KERNEL_W2), SEISMIC_DIM_V == 0 || PACKETS_FORM_INT8(KERNEL_W3)
#define PROJECTION_WEIGHT_TYPES Segments::Weights
#define PROJECTION_ROWS(g)                                                                                \
    Pro {                                                                                                 \
        Raw {                                                                                             \
            reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_HIDDEN)), SEISMIC_HIDDEN_STRIDE_0,  \
                SEISMIC_PTR(SEISMIC_BUFFER_INPUT_NORM), __uint_as_float((unsigned)SEISMIC_PARAM_EPSILON),  \
                SEISMIC_DIM_D, projection::AllRows{}                                                      \
        }                                                                                                 \
    }
#define PROJECTION_STAGING SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_STAGED)
#define PROJECTION_GROUPS SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_GROUPS)
#define PROJECTION_SEGMENTS                                                                               \
    Segments {                                                                                            \
        {SEGMENT_ROWS},                                                                                    \
            {KERNEL_W0_AT(SEISMIC_PTR(SEISMIC_BUFFER_QUERY_WEIGHT)),                                       \
             KERNEL_W1_AT(SEISMIC_PTR(SEISMIC_BUFFER_GATE_WEIGHT)),                                        \
             KERNEL_W2_AT(SEISMIC_PTR(SEISMIC_BUFFER_KEY_WEIGHT)),                                         \
             KERNEL_W3_AT(SEISMIC_PTR(SEISMIC_BUFFER_VALUE_WEIGHT))},                                      \
        {SEGMENT_OUT(SEISMIC_RESULT_0_OUT, SEISMIC_RESULT_0_STRIDE_0, SEISMIC_BUFFER_QUERY_SCALE,     \
                        SEISMIC_DIM_QS),                                                                  \
                SEGMENT_OUT(SEISMIC_RESULT_1_OUT, SEISMIC_RESULT_1_STRIDE_0, SEISMIC_BUFFER_GATE_SCALE, \
                            SEISMIC_DIM_GS),                                                              \
                SEGMENT_OUT(SEISMIC_RESULT_2_OUT, SEISMIC_RESULT_2_STRIDE_0, SEISMIC_BUFFER_KEY_SCALE,  \
                            SEISMIC_DIM_KS),                                                              \
                SEGMENT_OUT(SEISMIC_RESULT_3_OUT, SEISMIC_RESULT_3_STRIDE_0, SEISMIC_BUFFER_VALUE_SCALE, \
                            SEISMIC_DIM_VS)}                                                              \
    }

#ifdef SEISMIC_FORMING_ATTENTION_PROJECT_STAGE
template <unsigned INT8, unsigned OPERAND>
__global__ void attention_project_stage(SEISMIC_KERNEL_PARAMS) {
    PROJECTION_STAGE;
}
#endif

#ifdef SEISMIC_FORMING_ATTENTION_PROJECT_GEMV
template <unsigned STAGING, unsigned KSPLIT>
__global__ void attention_project_gemv(SEISMIC_KERNEL_PARAMS) {
    extern __shared__ uint4 dynamic_shared[];
    PROJECTION_SEGMENTED_GEMV(1, KSPLIT, STAGING, reinterpret_cast<projection::u8 *>(dynamic_shared));
}
#endif

#ifdef SEISMIC_FORMING_ATTENTION_PROJECT_GEMV16
template <unsigned STAGING, unsigned KSPLIT>
__global__ void attention_project_gemv16(SEISMIC_KERNEL_PARAMS) {
    PROJECTION_SEGMENTED_GEMV(2, KSPLIT, STAGING, nullptr);
}
#endif

#ifdef SEISMIC_FORMING_ATTENTION_PROJECT_GEMM_SMALL
template <unsigned INT8, unsigned OPERAND>
__global__ void attention_project_gemm_small(SEISMIC_KERNEL_PARAMS) {
    PROJECTION_SEGMENTED_GEMM(projection::SmallGemm, PROJECTION_OPERANDS);
}
#endif

#ifdef SEISMIC_FORMING_ATTENTION_PROJECT_GEMM
template <unsigned INT8, unsigned OPERAND, unsigned ROTATE, unsigned BLOCKS>
__launch_bounds__(256, BLOCKS) __global__ void attention_project_gemm(SEISMIC_KERNEL_PARAMS) {
    PROJECTION_SEGMENTED_GEMM(projection::LargeGemm<ROTATE>, PROJECTION_OPERANDS);
}
#endif
