// readout_project_rows: the vocabulary projection of already-normalized
// feature rows (the draft head's readout) into F32 logits, scaled by a
// present `weight_scale` (static WS = 1) and softcapped from the accumulator
// when `softcap` > 0. Where I is 0 a row's logits are the first S vocabulary
// rows', through the `dense` projection (single, `lib/projection/operator.cuh`);
// where I is 1 they are the S rows `token_rows` selects for it, through the
// routed projection's `choice` (`gather_choice`): each feature row an expert
// of one choice, the rows it selects that expert's matrix
// (`projection::Gathered`), a selection past the vocabulary -inf.
#define KERNEL_A SEISMIC_ELEMENT_A
#define KERNEL_W0 SEISMIC_WEIGHT
#include "lib/projection/operator.cuh"
#include "lib/routed/decode.cuh"

// The feature rows, read in place unless the weight takes them as NVFP4
// operands, which the GEMV forms at M = 1 and the stage launch above.
using Element = ELEMENT_OF(SEISMIC_ELEMENT_A);
using Raw = projection::Plain<Element, projection::AllRows>;
using Pro = projection::Operands<projection::OperandRecipe<packets::W0_OPERANDS_NVFP4>::NVFP4, Raw>;
using Scaling = projection::scaling<(SEISMIC_DIM_WS != 0)>;
using Epi = Scaling::type<projection::Logits>;

// The logits of feature row 0 at `out`: scaled, then softcapped.
#define READOUT_LOGITS(out)                                                                                    \
    Scaling::wrap(projection::Logits{(out), SEISMIC_RESULT_0_STRIDE_0, __uint_as_float((unsigned)SEISMIC_PARAM_SOFTCAP)}, \
                  projection::scale_factor(SEISMIC_PTR(SEISMIC_BUFFER_WEIGHT_SCALE), SEISMIC_DIM_WS, 0, 0), 1.0f)

#define PROJECTION_M SEISMIC_DIM_O
#define PROJECTION_K SEISMIC_DIM_D
#define PROJECTION_N SEISMIC_DIM_S
#define PROJECTION_G 1
#define PROJECTION_WA packets::W0
#define PROJECTION_WB projection::NoWeight
#define PROJECTION_WEIGHT_TYPES projection::op::Types<packets::W0>
#define PROJECTION_INT8 PACKETS_FORM_INT8(KERNEL_W0)
#define PROJECTION_ROWS(g) Pro{Raw{SEISMIC_PTR(SEISMIC_BUFFER_FEATURES), SEISMIC_FEATURES_STRIDE_0, projection::AllRows{}}}
#define PROJECTION_PLACE(g) SEISMIC_PTR(SEISMIC_BUFFER_FEATURES)
#define PROJECTION_PLACE_STRIDE SEISMIC_FEATURES_STRIDE_0
#define PROJECTION_STAGING SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_STAGED)
#define PROJECTION_GROUPS SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_GROUPS)
#define PROJECTION_PARTIALS(g, parts) reinterpret_cast<float *>(SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_PARTIALS))
#define PROJECTION_WEIGHTS(g) KERNEL_W0_AT(SEISMIC_PTR(SEISMIC_BUFFER_WEIGHT)), projection::NoWeight{}
#define PROJECTION_EPILOGUE(g) READOUT_LOGITS(SEISMIC_RESULT_0_OUT)

#ifdef SEISMIC_FORMING_READOUT_PROJECT_ROWS_STAGE
template <unsigned INT8, unsigned OPERAND>
__global__ void readout_project_rows_stage(SEISMIC_KERNEL_PARAMS) {
    PROJECTION_STAGE;
}
#endif

#ifdef SEISMIC_FORMING_READOUT_PROJECT_ROWS_GEMV
template <unsigned INT8, unsigned STAGING, unsigned KSPLIT, unsigned RESIDENCY>
__launch_bounds__(256, PROJECTION_GEMV_BLOCKS(KSPLIT, RESIDENCY)) __global__ void readout_project_rows_gemv(SEISMIC_KERNEL_PARAMS) {
    PROJECTION_GEMV(1, KSPLIT, STAGING, PROJECTION_GEMV_GRID(KSPLIT, RESIDENCY), RESIDENCY);
}
#endif

#ifdef SEISMIC_FORMING_READOUT_PROJECT_ROWS_GEMV16
template <unsigned INT8, unsigned STAGING, unsigned KSPLIT>
__global__ void readout_project_rows_gemv16(SEISMIC_KERNEL_PARAMS) {
    PROJECTION_GEMV(2, KSPLIT, STAGING, 0, 1);
}
#endif

#ifdef SEISMIC_FORMING_READOUT_PROJECT_ROWS_GEMM_SMALL
template <unsigned INT8, unsigned OPERAND, unsigned SPLIT>
__global__ void readout_project_rows_gemm_small(SEISMIC_KERNEL_PARAMS) {
    PROJECTION_GEMM(projection::SmallGemm, PROJECTION_OPERANDS);
}
#endif

#ifdef SEISMIC_FORMING_READOUT_PROJECT_ROWS_GEMM
template <unsigned INT8, unsigned OPERAND, unsigned SPLIT, unsigned ROTATE, unsigned BLOCKS>
__launch_bounds__(256, BLOCKS) __global__ void readout_project_rows_gemm(SEISMIC_KERNEL_PARAMS) {
    PROJECTION_GEMM(projection::LargeGemm<ROTATE>, PROJECTION_OPERANDS);
}
#endif

#ifdef SEISMIC_FORMING_READOUT_PROJECT_ROWS_FINALIZE
template <unsigned SPLIT>
__global__ void readout_project_rows_finalize(SEISMIC_KERNEL_PARAMS) {
    PROJECTION_FINALIZE;
}
#endif

#ifdef SEISMIC_FORMING_READOUT_PROJECT_ROWS_GATHER_CHOICE
// A selection's logit, or -inf for a selection past the vocabulary (which
// the GEMV read as row 0).
struct SelectedLogits {
    Epi logits;
    const int *selected;
    projection::u64 stride;
    element::Out<element::F32> out;
    __device__ __forceinline__ void operator()(unsigned m, projection::u64 n, float first, float second) const {
        if ((projection::u64)(long long)selected[n * stride] < SEISMIC_DIM_V)
            logits(m, n, first, second);
        else
            out.put(n, __int_as_float(0xff800000));
    }
};

// Block row y = feature row m's choice (of I): the routed decode GEMV of its
// one row over the S weight rows it selects, a 16-row tile of them a warp
// group's, the row formed in the block's shared memory (`shared_row`).
template <int GATHER_TPW, int GATHER_KSPLIT>
__launch_bounds__(routed::DECODE_THREADS) __global__ void readout_project_rows_gather_choice(SEISMIC_KERNEL_PARAMS) {
    using Shape = routed::DecodeShape<GATHER_TPW, GATHER_KSPLIT>;
    __shared__ projection::GemvShared<Shape, projection::SharedRow> shared;
    const projection::u64 m = blockIdx.y / SEISMIC_DIM_I, choice = blockIdx.y % SEISMIC_DIM_I;
    // Block-collective: before any warp leaves.
    const projection::SharedRow pro = projection::shared_row(
        Pro{Raw{SEISMIC_PTR(SEISMIC_BUFFER_FEATURES) + m * SEISMIC_FEATURES_STRIDE_0 * Element::bytes, SEISMIC_FEATURES_STRIDE_0,
                projection::AllRows{}}},
        SEISMIC_DIM_D);
    const int *selected = reinterpret_cast<const int *>(SEISMIC_PTR(SEISMIC_BUFFER_TOKEN_ROWS)) +
                          m * SEISMIC_TOKEN_ROWS_STRIDE_0 + choice * SEISMIC_TOKEN_ROWS_STRIDE_1;
    const element::Out<element::F32> row = SEISMIC_RESULT_0_OUT.offset(m * SEISMIC_RESULT_0_STRIDE_0);
    const SelectedLogits epi{READOUT_LOGITS(row), selected, SEISMIC_TOKEN_ROWS_STRIDE_2, row};
    const projection::Gathered<packets::W0> weight{KERNEL_W0_AT(SEISMIC_PTR(SEISMIC_BUFFER_WEIGHT)), selected,
                                                   SEISMIC_TOKEN_ROWS_STRIDE_2, SEISMIC_DIM_S, SEISMIC_DIM_V};
    projection::gemv_segment<Shape>(shared, pro, 1u, SEISMIC_DIM_D / 64, Shape::tile_group(), SEISMIC_DIM_S, weight,
                                    projection::NoWeight{}, epi);
}
#endif
