// readout_planes_rows: final RMS prologue over the `out_rows` hidden rows and
// the exact vocabulary projection of a progressive head into F32 logits: the
// `dense` projection (single, `lib/projection/operator.h`) over the packets
// of the head's exact view (`lib/readout/progressive.h`), whose GEMV is the
// exact level's own recipe. Each threadgroup normalizes its rows in place to
// 16 rows; past them `stage` normalizes them once into `normalized`, which
// the staged tiles read.
#define KERNEL_A SEISMIC_ELEMENT_A
#include "lib/readout/progressive.h"
#include "lib/projection/operator.h"
#if defined(SEISMIC_FORMING_READOUT_PLANES_ROWS_GEMV) || defined(SEISMIC_FORMING_READOUT_PLANES_ROWS_GEMV_ROWS) || defined(SEISMIC_FORMING_READOUT_PLANES_ROWS_GEMV_PAIR)
#include "lib/projection/gemv.h"
#endif

// The plane set's weight: its exact view.
PACKETS_BIND(W0, readout::ProgressiveExact)

typedef progressive::activation activation_element;
typedef ELEMENT_OF(SEISMIC_NORM) norm_element;
constant constexpr bool NVFP4 = false;
typedef projection::operands<NVFP4> Operands;

#define PROJECTION_ARGUMENTS                                                            \
    device const float *hidden [[buffer(SEISMIC_BUFFER_HIDDEN)]],                       \
    device const uchar *norm [[buffer(SEISMIC_BUFFER_NORM)]],                           \
    device const uint *top [[buffer(SEISMIC_BUFFER_TOP)]],                              \
    device const uint *bit3 [[buffer(SEISMIC_BUFFER_BIT3)]],                            \
    device const uint *rest [[buffer(SEISMIC_BUFFER_REST)]],                            \
    device const half *scales [[buffer(SEISMIC_BUFFER_SCALES)]],                        \
    device const int *out_rows [[buffer(SEISMIC_BUFFER_OUT_ROWS)]],                     \
    SEISMIC_RESULT_0_PARAM(logits),                                                     \
    device uchar *normalized [[buffer(SEISMIC_BUFFER_SCRATCH_NORMALIZED)]],             \
    constant ulong *seismic_words [[buffer(SEISMIC_BUFFER_WORDS)]]

// `raw` is the selected hidden rows under the norm prologue; `in` the rows
// `stage` normalizes, which the staged tiles read.
#define PROJECTION_OPERANDS(group)                                                      \
    const uint m = uint(SEISMIC_DIM_O), n = uint(SEISMIC_DIM_V), k = uint(SEISMIC_DIM_D); \
    projection::InputOf<activation_element, norm_element, projection::SelectedRows, SEISMIC_DIM_NM, SEISMIC_DIM_PT>::type \
        raw{hidden, SEISMIC_HIDDEN_STRIDE_0, SEISMIC_HIDDEN_STRIDE_2, norm, SEISMIC_NORM_STRIDE_0, \
        as_type<float>(uint(SEISMIC_PARAM_EPSILON)), k, {out_rows}};                    \
    const projection::Plain<activation_element, projection::AllRows> in{normalized, k, 1, k, {}}; \
    const projection::Logits out{element::Out<element::F32>(logits), SEISMIC_RESULT_0_STRIDE_0, SEISMIC_RESULT_0_STRIDE_1, \
        as_type<float>(uint(SEISMIC_PARAM_SOFTCAP))};                                   \
    const projection::Weights<packets::W0> w{top, SEISMIC_TOP_STRIDE_0, bit3, SEISMIC_BIT3_STRIDE_0, rest, \
        SEISMIC_REST_STRIDE_0, SEISMIC_REST_STRIDE_1, SEISMIC_REST_STRIDE_2, scales, SEISMIC_SCALES_STRIDE_0}

// The GEMV's and the batched launch's rows: the norm applied as they stage,
// from their sums of squares over the threadgroup.
#define PROJECTION_GROUPS 1
#define PROJECTION_GEMV_INPUT(x)                                                        \
    PROJECTION_SQUARES_SHARED(squares, decltype(raw)::parts);                           \
    projection::threadgroup_squares_runtime(raw, m, squares, simdgroups, sg, lane);     \
    const auto x = projection::shared_norm(raw, squares)

// The normalized rows into `normalized`, one row a threadgroup.
#ifdef SEISMIC_FORMING_READOUT_PLANES_ROWS_STAGE
kernel void readout_planes_rows_stage(PROJECTION_ARGUMENTS,
    uint item [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]]) {
    PROJECTION_NORMALIZE_SHARED(norms);
    PROJECTION_OPERANDS(0u);
    projection::device_normalize<256>(raw, item, normalized, k, norms, thread_index);
}
#endif

// The `dense` projection's launches (`lib/projection/operator.h`).
#ifdef SEISMIC_FORMING_READOUT_PLANES_ROWS_GEMV
template <uint MATRIX, uint LANES, uint TILED, uint STAGE, uint ROWS>
kernel void readout_planes_rows_gemv(PROJECTION_ARGUMENTS, PROJECTION_ROW_ARGUMENTS) {
    PROJECTION_SINGLE_GEMV(ONE, TILED);
}
#endif
#ifdef SEISMIC_FORMING_READOUT_PLANES_ROWS_GEMV_PAIR
template <uint MATRIX, uint LANES, uint TILED, uint STAGE, uint ROWS>
kernel void readout_planes_rows_gemv_pair(PROJECTION_ARGUMENTS, PROJECTION_ROW_ARGUMENTS) {
    PROJECTION_SINGLE_GEMV(PAIR, TILED);
}
#endif
#ifdef SEISMIC_FORMING_READOUT_PLANES_ROWS_GEMV_ROWS
template <uint MATRIX, uint LANES, uint TILED, uint STAGE, uint ROWS>
kernel void readout_planes_rows_gemv_rows(PROJECTION_ARGUMENTS, PROJECTION_ROW_ARGUMENTS) {
    PROJECTION_SINGLE_GEMV(SEVERAL, TILED);
}
#endif
#ifdef SEISMIC_FORMING_READOUT_PLANES_ROWS_BATCH
template <uint MATRIX, uint BATCH_ROWS, uint BATCH_PARTS>
kernel void readout_planes_rows_batch(PROJECTION_ARGUMENTS, PROJECTION_ROW_ARGUMENTS) {
    PROJECTION_SINGLE_BATCH_FORMED;
}
#endif
#ifdef SEISMIC_FORMING_READOUT_PLANES_ROWS_GEMM
template <uint TILE_M, uint TILE_N>
kernel void readout_planes_rows_gemm(PROJECTION_ARGUMENTS, PROJECTION_TILE_ARGUMENTS) {
    PROJECTION_SINGLE_GEMM_WHOLE;
}
#endif
