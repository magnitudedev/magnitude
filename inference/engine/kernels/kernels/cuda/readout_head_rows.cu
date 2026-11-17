// readout_head_rows: final RMS prologue over the `out_rows` rows, then the
// vocabulary projection into F32 logits, scaled by a present `weight_scale`
// (static WS = 1) and softcapped from the accumulator when `softcap` > 0: the
// forms of the `dense` projection (`lib/projection/operator.cuh`).
#define KERNEL_W0 SEISMIC_WEIGHT
#include "lib/projection/operator.cuh"


using Raw = projection::Input<ELEMENT_OF(SEISMIC_NORM), projection::SelectedRows, SEISMIC_DIM_NM, SEISMIC_DIM_PT>;
using Pro = projection::Operands<projection::OperandRecipe<packets::W0_OPERANDS_NVFP4>::NVFP4, Raw>;
using Scaling = projection::scaling<(SEISMIC_DIM_WS != 0)>;

#define PROJECTION_M SEISMIC_DIM_O
#define PROJECTION_K SEISMIC_DIM_D
#define PROJECTION_N SEISMIC_DIM_V
#define PROJECTION_G 1
#define PROJECTION_WA packets::W0
#define PROJECTION_WB projection::NoWeight
#define PROJECTION_WEIGHT_TYPES projection::op::Types<packets::W0>
#define PROJECTION_INT8 PACKETS_FORM_INT8(KERNEL_W0)
#define PROJECTION_ROWS(g)                                                                                        \
    Pro {                                                                                                         \
        Raw {                                                                                                     \
            reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_HIDDEN)), SEISMIC_HIDDEN_STRIDE_0,          \
                SEISMIC_PTR(SEISMIC_BUFFER_NORM), __uint_as_float((unsigned)SEISMIC_PARAM_EPSILON), SEISMIC_DIM_D, \
                projection::SelectedRows{reinterpret_cast<const int *>(SEISMIC_PTR(SEISMIC_BUFFER_OUT_ROWS))}     \
        }                                                                                                         \
    }
#define PROJECTION_PLACE(g) SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_STAGED)
#define PROJECTION_PLACE_STRIDE SEISMIC_DIM_D
#define PROJECTION_STAGING SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_STAGED)
#define PROJECTION_GROUPS SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_GROUPS)
#define PROJECTION_PARTIALS(g, parts) reinterpret_cast<float *>(SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_PARTIALS))
#define PROJECTION_WEIGHTS(g) KERNEL_W0_AT(SEISMIC_PTR(SEISMIC_BUFFER_WEIGHT)), projection::NoWeight{}
#define PROJECTION_EPILOGUE(g)                                                                                \
    Scaling::wrap(projection::Logits{SEISMIC_RESULT_0_OUT, SEISMIC_RESULT_0_STRIDE_0,              \
                                     __uint_as_float((unsigned)SEISMIC_PARAM_SOFTCAP)},                           \
                  projection::scale_factor(SEISMIC_PTR(SEISMIC_BUFFER_WEIGHT_SCALE), SEISMIC_DIM_WS, 0, 0), 1.0f)

#ifdef SEISMIC_FORMING_READOUT_HEAD_ROWS_STAGE
template <unsigned INT8, unsigned OPERAND>
__global__ void readout_head_rows_stage(SEISMIC_KERNEL_PARAMS) {
    PROJECTION_STAGE;
}
#endif

#ifdef SEISMIC_FORMING_READOUT_HEAD_ROWS_GEMV
template <unsigned STAGING, unsigned KSPLIT, unsigned RESIDENCY>
__launch_bounds__(256, PROJECTION_GEMV_BLOCKS(KSPLIT, RESIDENCY)) __global__ void readout_head_rows_gemv(SEISMIC_KERNEL_PARAMS) {
    PROJECTION_GEMV(1, KSPLIT, STAGING, PROJECTION_GEMV_GRID(KSPLIT, RESIDENCY), RESIDENCY);
}
#endif

#ifdef SEISMIC_FORMING_READOUT_HEAD_ROWS_GEMV16
template <unsigned STAGING, unsigned KSPLIT>
__global__ void readout_head_rows_gemv16(SEISMIC_KERNEL_PARAMS) {
    PROJECTION_GEMV(2, KSPLIT, STAGING, 0, 1);
}
#endif

#ifdef SEISMIC_FORMING_READOUT_HEAD_ROWS_GEMM_SMALL
template <unsigned INT8, unsigned OPERAND, unsigned SPLIT>
__global__ void readout_head_rows_gemm_small(SEISMIC_KERNEL_PARAMS) {
    PROJECTION_GEMM(projection::SmallGemm, PROJECTION_OPERANDS);
}
#endif

#ifdef SEISMIC_FORMING_READOUT_HEAD_ROWS_GEMM
template <unsigned INT8, unsigned OPERAND, unsigned SPLIT, unsigned ROTATE, unsigned BLOCKS>
__launch_bounds__(256, BLOCKS) __global__ void readout_head_rows_gemm(SEISMIC_KERNEL_PARAMS) {
    PROJECTION_GEMM(projection::LargeGemm<ROTATE>, PROJECTION_OPERANDS);
}
#endif

#ifdef SEISMIC_FORMING_READOUT_HEAD_ROWS_FINALIZE
template <unsigned SPLIT>
__global__ void readout_head_rows_finalize(SEISMIC_KERNEL_PARAMS) {
    PROJECTION_FINALIZE;
}
#endif
