// readout_top_rows: level one of a certified selection (`readout.seismic`),
// in three launches. `stage` forms each row's final norm into `features` and
// its length |x| over those rounded values into `lengths`, once. The `dense`
// projection (single, `lib/projection/operator.h`) then projects `features`
// over the packets of the 4-bit view of a progressive head
// (`lib/readout/progressive.h`) with the bound epilogue: each vocabulary
// row's lower-bound score raises its output row's key (a threadgroup atomic
// maximum, then one device maximum a threadgroup). `publish` turns the keys
// into the thresholds and clears them.
//
// The head's grid is a vocabulary of threadgroups, so whatever one of them
// does besides streaming its weight rows is paid once per threadgroup: the
// rows are normalized and measured once in `stage` rather than in every
// threadgroup before its first weight load, and the thresholds are published
// once in `publish` rather than by the last threadgroup to arrive, which
// would hold every threadgroup on its stores and an arrival count.
#define KERNEL_A SEISMIC_ELEMENT_A
#include "lib/readout/progressive.h"
#include "lib/projection/operator.h"
#if defined(SEISMIC_FORMING_READOUT_TOP_ROWS_GEMV) || defined(SEISMIC_FORMING_READOUT_TOP_ROWS_GEMV_ROWS) || defined(SEISMIC_FORMING_READOUT_TOP_ROWS_GEMV_PAIR)
#include "lib/projection/gemv.h"
#endif

// The plane set's weight: its 4-bit view.
PACKETS_BIND(W0, readout::ProgressiveTop)

typedef progressive::activation activation_element;
typedef ELEMENT_OF(SEISMIC_NORM) norm_element;
constant constexpr bool NVFP4 = false;
typedef projection::operands<NVFP4> Operands;

// The view logits to `logits`; each competing row's lower-bound score into
// `lowest`. A lane loads the selection side of the output it emits with its
// first packet (`early`), so the weight stream, not those loads, ends its
// threadgroup.
struct BoundOut {
    element::Out<element::F32> logits;
    ulong row, col;
    device const float *radius;
    ulong radius_stride;
    device const float *lengths;
    ulong length_stride;
    threadgroup atomic_uint *lowest;
    progressive::Selection selection;
    struct early {
        bool competes;
        float radius, length;
    };
    early load_early(uint m, uint n) const {
        return {selection.competes(m, n), radius[ulong(n) * radius_stride], lengths[ulong(m) * length_stride]};
    }
    void store(uint m, uint n, float value, thread const early &e) const {
        logits.put(ulong(m) * row + ulong(n) * col, value);
        if (e.competes)
            selection.raise(&lowest[m], m, n, value - e.radius * e.length);
    }
    void store(uint m, uint n, float value) const { store(m, n, value, load_early(m, n)); }
    void store2(uint m, uint n, float first, float second) const {
        store(m, n, first);
        store(m, n + 1u, second);
    }
};

namespace projection {
template <>
struct early<BoundOut> {
    typedef BoundOut::early values;
    static values load(thread const BoundOut &out, uint m, uint n) { return out.load_early(m, n); }
    static void store(thread const BoundOut &out, uint m, uint n, float value, thread const values &e) {
        out.store(m, n, value, e);
    }
};
} // namespace projection

#define PROJECTION_ARGUMENTS                                                            \
    device const float *hidden [[buffer(SEISMIC_BUFFER_HIDDEN)]],                       \
    device const uchar *norm [[buffer(SEISMIC_BUFFER_NORM)]],                           \
    device const uint *top [[buffer(SEISMIC_BUFFER_TOP)]],                              \
    device const half *scales [[buffer(SEISMIC_BUFFER_SCALES)]],                        \
    device const float *radius [[buffer(SEISMIC_BUFFER_RADIUS)]],                       \
    device const int *out_rows [[buffer(SEISMIC_BUFFER_OUT_ROWS)]],                     \
    device const uint *draws [[buffer(SEISMIC_BUFFER_DRAWS)]],                          \
    device const float *temperature [[buffer(SEISMIC_BUFFER_TEMPERATURE)]],             \
    device const uint *mask [[buffer(SEISMIC_BUFFER_MASK)]],                            \
    device const int *constrained [[buffer(SEISMIC_BUFFER_CONSTRAINED)]],               \
    SEISMIC_RESULT_0_PARAM(logits),                                                     \
    SEISMIC_RESULT_1_PARAM(threshold),                                                  \
    SEISMIC_RESULT_2_PARAM(features),                                                   \
    SEISMIC_RESULT_3_PARAM(lengths),                                                    \
    device atomic_uint *bounds [[buffer(SEISMIC_BUFFER_SCRATCH_BOUNDS)]],               \
    constant ulong *seismic_words [[buffer(SEISMIC_BUFFER_WORDS)]]

// The rows' lengths as `stage` publishes them.
#define TOP_ROWS_LENGTHS reinterpret_cast<device const float *>(element::Out<element::F32>(lengths).stored())

// Output row `item`: its final norm into `features` and its length |x| over
// the rounded values into `lengths`. The row's square sum is reduced in the
// GEMV prologue's fixed parts (`threadgroup_squares_runtime`), so `features`
// holds exactly the values the GEMV would form from the hidden row itself.
#ifdef SEISMIC_FORMING_READOUT_TOP_ROWS_STAGE
kernel void readout_top_rows_stage(PROJECTION_ARGUMENTS,
    uint item [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint simdgroups [[simdgroups_per_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    const uint d = uint(SEISMIC_DIM_D);
    const projection::InputOf<activation_element, norm_element, projection::SelectedRows, SEISMIC_DIM_NM, SEISMIC_DIM_PT>::type
        raw{hidden, SEISMIC_HIDDEN_STRIDE_0, SEISMIC_HIDDEN_STRIDE_2, norm, SEISMIC_NORM_STRIDE_0,
            as_type<float>(uint(SEISMIC_PARAM_EPSILON)), d, {out_rows + item}};
    PROJECTION_SQUARES_SHARED(squares, decltype(raw)::parts);
    threadgroup float parts[32];
    projection::threadgroup_squares_runtime(raw, 1u, squares, simdgroups, sg, lane);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const auto x = projection::shared_norm(raw, squares);
    const element::Out<activation_element> rows(features);
    const uint threads = simdgroups * 32u;
    float sum = 0.0f;
    for (uint i = 8u * tid; i < d; i += 8u * threads) {
        float4 even, odd;
        x.load8(0u, i, 0.0f, even, odd);
        for (uint j = 0; j < 4u; ++j) {
            const float a = activation_element::round(even[j]);
            const float b = activation_element::round(odd[j]);
            sum = metal::fma(a, a, sum);
            sum = metal::fma(b, b, sum);
        }
        rows.put8(ulong(item) * d + i, even, odd);
    }
    sum = simd_sum(sum);
    if (lane == 0)
        parts[sg] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float total = 0.0f;
        for (uint s = 0; s < simdgroups; ++s)
            total += parts[s];
        element::Out<element::F32>(lengths).put(ulong(item) * SEISMIC_RESULT_3_STRIDE_0, metal::sqrt(total));
    }
}
#endif

// `x` is the staged features; `out` the bound epilogue over the rows'
// lengths and this threadgroup's keys, which the GEMV's opening barrier
// publishes cleared.
#define PROJECTION_OPERANDS(group)                                                      \
    const uint m = uint(SEISMIC_DIM_O), n = uint(SEISMIC_DIM_V), k = uint(SEISMIC_DIM_D); \
    const projection::Plain<activation_element, projection::AllRows> staged{            \
        element::Out<activation_element>(features).stored(), k, 1, k, {}};              \
    const projection::Weights<packets::W0> w{top, SEISMIC_TOP_STRIDE_0, scales, SEISMIC_SCALES_STRIDE_0}; \
    threadgroup atomic_uint lowest[SEISMIC_CONST_VERIFY_ROWS];                          \
    /* Radius column 0: the 4-bit view's. */                                            \
    const BoundOut out{element::Out<element::F32>(logits), SEISMIC_RESULT_0_STRIDE_0, SEISMIC_RESULT_0_STRIDE_1, radius, \
        SEISMIC_RADIUS_STRIDE_0, TOP_ROWS_LENGTHS, SEISMIC_RESULT_3_STRIDE_0, lowest,   \
        {draws, SEISMIC_DRAWS_STRIDE_0, SEISMIC_DRAWS_STRIDE_1, temperature, SEISMIC_TEMPERATURE_STRIDE_0, mask, \
            SEISMIC_MASK_STRIDE_0, SEISMIC_MASK_STRIDE_1, constrained, SEISMIC_CONSTRAINED_STRIDE_0, \
            as_type<float>(uint(SEISMIC_PARAM_SOFTCAP))}}

#define PROJECTION_GROUPS 1
#define PROJECTION_GEMV_INPUT(x)                                                        \
    if (tid < SEISMIC_CONST_VERIFY_ROWS)                                                \
        atomic_store_explicit(&lowest[tid], 0u, memory_order_relaxed);                  \
    const auto x = staged

// This threadgroup's keys into the launch's (`bounds`, sync scratch).
#define TOP_ROWS_RAISE                                                                  \
    threadgroup_barrier(mem_flags::mem_threadgroup);                                    \
    if (tid < m)                                                                        \
        atomic_fetch_max_explicit(&bounds[tid], atomic_load_explicit(&lowest[tid], memory_order_relaxed), \
            memory_order_relaxed)

// The `dense` projection's launches (`lib/projection/operator.h`).
#ifdef SEISMIC_FORMING_READOUT_TOP_ROWS_GEMV
template <uint MATRIX_FROM, uint LANES, uint TILED, uint STAGE, uint ROWS>
kernel void readout_top_rows_gemv(PROJECTION_ARGUMENTS, PROJECTION_ROW_ARGUMENTS,
    uint tid [[thread_index_in_threadgroup]]) {
    PROJECTION_SINGLE_GEMV(ONE, TILED);
    TOP_ROWS_RAISE;
}
#endif
#ifdef SEISMIC_FORMING_READOUT_TOP_ROWS_GEMV_PAIR
template <uint MATRIX_FROM, uint LANES, uint TILED, uint STAGE, uint ROWS>
kernel void readout_top_rows_gemv_pair(PROJECTION_ARGUMENTS, PROJECTION_ROW_ARGUMENTS,
    uint tid [[thread_index_in_threadgroup]]) {
    PROJECTION_SINGLE_GEMV(PAIR, TILED);
    TOP_ROWS_RAISE;
}
#endif
#ifdef SEISMIC_FORMING_READOUT_TOP_ROWS_GEMV_ROWS
template <uint MATRIX_FROM, uint LANES, uint TILED, uint STAGE, uint ROWS>
kernel void readout_top_rows_gemv_rows(PROJECTION_ARGUMENTS, PROJECTION_ROW_ARGUMENTS,
    uint tid [[thread_index_in_threadgroup]]) {
    PROJECTION_SINGLE_GEMV(SEVERAL, TILED);
    TOP_ROWS_RAISE;
}
#endif

// Each row's threshold from its key (-inf where no row raised it, or where
// the row's length is not finite: no bound holds); the keys cleared, so the
// next launch over `bounds` starts from zero.
#ifdef SEISMIC_FORMING_READOUT_TOP_ROWS_PUBLISH
kernel void readout_top_rows_publish(PROJECTION_ARGUMENTS, uint tid [[thread_index_in_threadgroup]]) {
    if (tid >= uint(SEISMIC_DIM_O))
        return;
    const uint key = atomic_load_explicit(&bounds[tid], memory_order_relaxed);
    const float raised = key == 0u ? -INFINITY : order::value(key);
    const float length = TOP_ROWS_LENGTHS[ulong(tid) * SEISMIC_RESULT_3_STRIDE_0];
    element::Out<element::F32>(threshold).put(ulong(tid) * SEISMIC_RESULT_1_STRIDE_0,
        metal::isfinite(length) ? raised : -INFINITY);
    atomic_store_explicit(&bounds[tid], 0u, memory_order_relaxed);
}
#endif
