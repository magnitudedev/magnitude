// attention_project: the query, gate, key and value projections of the
// normalized input rows, the forms of the `dense` projection
// (`lib/projection/operator.cuh`), segmented. A call in mode 1 zeroes the
// query and gate segments.
#define KERNEL_A SEISMIC_ELEMENT_A
#define KERNEL_W0 SEISMIC_QUERY_WEIGHT
#define KERNEL_W1 SEISMIC_GATE_WEIGHT
#define KERNEL_W2 SEISMIC_KEY_WEIGHT
#define KERNEL_W3 SEISMIC_VALUE_WEIGHT
#define KERNEL_W4 SEISMIC_INDEXER_QUERY_WEIGHT
#define KERNEL_W5 SEISMIC_INDEXER_KEY_WEIGHT
#include "lib/projection/operator.cuh"

SEISMIC_PROGRAMMATIC_DEPENDENCY

using Raw = projection::Input<ELEMENT_OF(SEISMIC_INPUT_NORM), projection::AllRows, SEISMIC_DIM_NM, SEISMIC_DIM_PT>;
using Pro = projection::Operands<
    projection::OperandRecipe<packets::W0_OPERANDS_NVFP4, packets::W1_OPERANDS_NVFP4, packets::W2_OPERANDS_NVFP4,
                              packets::W3_OPERANDS_NVFP4, packets::W4_OPERANDS_NVFP4, packets::W5_OPERANDS_NVFP4>::NVFP4,
    Raw>;
// A segment's result, after its weight's second-level scale where any
// segment has a scale port (an absent port's factor is 1), published in Y
// (segments 0..4) or IY (segment 5, indexer key).
using Store = projection::Store<ELEMENT_OF(SEISMIC_ELEMENT_Y)>;
using StoreIK = projection::Store<ELEMENT_OF(SEISMIC_ELEMENT_IY)>;
using Scaling = projection::scaling<(SEISMIC_DIM_QS != 0 || SEISMIC_DIM_GS != 0 || SEISMIC_DIM_KS != 0 ||
                                     SEISMIC_DIM_VS != 0)>;
using Out = Scaling::type<Store>;

// The six segments: rows, weights and results in segment order.
#define SEGMENT_ROWS SEISMIC_DIM_Q, SEISMIC_DIM_GR, SEISMIC_DIM_K, SEISMIC_DIM_V, SEISMIC_DIM_IQ, SEISMIC_DIM_IK
using Segments = projection::op::SegmentSet<
    1, projection::op::Types<packets::W0, packets::W1, packets::W2, packets::W3, packets::W4, packets::W5>,
    projection::op::Types<Out, Out, Out, Out, Store, StoreIK>, projection::op::empty_mask(SEGMENT_ROWS)>;

#define SEGMENT_OUT(STORE, RESULT, STRIDE, SCALE, EXTENT)                                                 \
    Scaling::wrap(STORE{RESULT, STRIDE, 0}, projection::scale_factor(SEISMIC_PTR(SCALE), EXTENT, 0, 0),   \
                  1.0f)

#define PROJECTION_M SEISMIC_DIM_M
#define PROJECTION_K SEISMIC_DIM_D
#define PROJECTION_ZERO SEISMIC_PARAM_PROJECT_MODE
// An empty segment's binding is irrelevant to the INT8 path.
#define PROJECTION_INT8                                                                                      \
    SEISMIC_DIM_Q == 0 || PACKETS_FORM_INT8(KERNEL_W0), SEISMIC_DIM_GR == 0 || PACKETS_FORM_INT8(KERNEL_W1), \
        SEISMIC_DIM_K == 0 || PACKETS_FORM_INT8(KERNEL_W2), SEISMIC_DIM_V == 0 || PACKETS_FORM_INT8(KERNEL_W3), \
        SEISMIC_DIM_IQ == 0 || PACKETS_FORM_INT8(KERNEL_W4), SEISMIC_DIM_IK == 0 || PACKETS_FORM_INT8(KERNEL_W5)
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
             KERNEL_W3_AT(SEISMIC_PTR(SEISMIC_BUFFER_VALUE_WEIGHT)),                                       \
             KERNEL_W4_AT(SEISMIC_PTR(SEISMIC_BUFFER_INDEXER_QUERY_WEIGHT)),                               \
             KERNEL_W5_AT(SEISMIC_PTR(SEISMIC_BUFFER_INDEXER_KEY_WEIGHT))},                                \
        {SEGMENT_OUT(Store, SEISMIC_RESULT_0_OUT, SEISMIC_RESULT_0_STRIDE_0, SEISMIC_BUFFER_QUERY_SCALE,   \
                     SEISMIC_DIM_QS),                                                                      \
         SEGMENT_OUT(Store, SEISMIC_RESULT_1_OUT, SEISMIC_RESULT_1_STRIDE_0, SEISMIC_BUFFER_GATE_SCALE,    \
                     SEISMIC_DIM_GS),                                                                      \
         SEGMENT_OUT(Store, SEISMIC_RESULT_2_OUT, SEISMIC_RESULT_2_STRIDE_0, SEISMIC_BUFFER_KEY_SCALE,     \
                     SEISMIC_DIM_KS),                                                                      \
         SEGMENT_OUT(Store, SEISMIC_RESULT_3_OUT, SEISMIC_RESULT_3_STRIDE_0, SEISMIC_BUFFER_VALUE_SCALE,   \
                     SEISMIC_DIM_VS),                                                                      \
         Store{SEISMIC_RESULT_4_OUT, SEISMIC_RESULT_4_STRIDE_0, 0},                                        \
         StoreIK{SEISMIC_RESULT_5_OUT, SEISMIC_RESULT_5_STRIDE_0, 0}}                                      \
    }

#ifdef SEISMIC_FORMING_ATTENTION_PROJECT_STAGE
template <unsigned INT8, unsigned OPERAND>
__global__ void attention_project_stage(SEISMIC_KERNEL_PARAMS) {
    PROJECTION_STAGE;
}
#endif

#ifdef SEISMIC_FORMING_ATTENTION_PROJECT_GEMV
template <unsigned INT8, unsigned STAGING, unsigned KSPLIT>
__global__ void attention_project_gemv(SEISMIC_KERNEL_PARAMS) {
    PROJECTION_SEGMENTED_GEMV(1, KSPLIT, STAGING);
}
#endif

#ifdef SEISMIC_FORMING_ATTENTION_PROJECT_GEMV16
template <unsigned INT8, unsigned STAGING, unsigned KSPLIT>
__global__ void attention_project_gemv16(SEISMIC_KERNEL_PARAMS) {
    PROJECTION_SEGMENTED_GEMV(2, KSPLIT, STAGING);
}
#endif

#ifdef SEISMIC_FORMING_ATTENTION_PROJECT_GEMM_SMALL
template <unsigned INT8, unsigned OPERAND, unsigned GEMM_SPLIT>
__global__ void attention_project_gemm_small(SEISMIC_KERNEL_PARAMS) {
    PROJECTION_SEGMENTED_GEMM(projection::SmallGemm, PROJECTION_OPERANDS);
}
#endif

#ifdef SEISMIC_FORMING_ATTENTION_PROJECT_GEMM
template <unsigned INT8, unsigned OPERAND, unsigned GEMM_SPLIT, unsigned ROTATE, unsigned BLOCKS>
__launch_bounds__(256, BLOCKS) __global__ void attention_project_gemm(SEISMIC_KERNEL_PARAMS) {
    PROJECTION_SEGMENTED_GEMM(projection::LargeGemm<ROTATE>, PROJECTION_OPERANDS);
}
#endif
