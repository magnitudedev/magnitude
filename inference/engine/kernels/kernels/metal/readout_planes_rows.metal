// readout_planes_rows: final RMS prologue over the `out_rows` hidden rows and
// the exact vocabulary projection of a progressive head into F32 logits:
// `readout_head_rows`' launches over the projection library's packets of the
// head's exact view (`lib/readout/progressive.h`).
#include "lib/readout/progressive.h"
#include "lib/projection/gemv.h"

typedef progressive::activation activation;
typedef ELEMENT_OF(SEISMIC_NORM) norm_element;

#define PLANES_ROWS_ARGUMENTS                                                             \
    device const float *hidden [[buffer(SEISMIC_BUFFER_HIDDEN)]],                       \
    device const uchar *norm [[buffer(SEISMIC_BUFFER_NORM)]],                           \
    device const uint *top [[buffer(SEISMIC_BUFFER_TOP)]],                              \
    device const uint *bit3 [[buffer(SEISMIC_BUFFER_BIT3)]],                            \
    device const uint *rest [[buffer(SEISMIC_BUFFER_REST)]],                            \
    device const half *scales [[buffer(SEISMIC_BUFFER_SCALES)]],                        \
    device const int *out_rows [[buffer(SEISMIC_BUFFER_OUT_ROWS)]],                     \
    device uchar *logits [[buffer(SEISMIC_RESULT_0_BUFFER)]],                           \
    device uchar *normalized [[buffer(SEISMIC_BUFFER_SCRATCH_NORMALIZED)]],             \
    constant ulong *seismic_words [[buffer(SEISMIC_BUFFER_WORDS)]]

#define PLANES_ROWS_OPERANDS                                                              \
    const uint k = uint(SEISMIC_DIM_D);                                                 \
    projection::InputOf<activation, norm_element, projection::SelectedRows, SEISMIC_DIM_NM, SEISMIC_DIM_PT>::type \
        in{hidden, SEISMIC_HIDDEN_STRIDE_0, \
        SEISMIC_HIDDEN_STRIDE_2, norm, SEISMIC_NORM_STRIDE_0,                           \
        as_type<float>(uint(SEISMIC_PARAM_EPSILON)), k, {out_rows}};                    \
    const projection::Logits out{logits, SEISMIC_RESULT_0_STRIDE_0, SEISMIC_RESULT_0_STRIDE_1, as_type<float>(uint(SEISMIC_PARAM_SOFTCAP))}; \
    const projection::Weights<packets::ProgressiveExact> w{top, SEISMIC_TOP_STRIDE_0, bit3, \
        SEISMIC_BIT3_STRIDE_0, rest, SEISMIC_REST_STRIDE_0, SEISMIC_REST_STRIDE_1, SEISMIC_REST_STRIDE_2, \
        scales, SEISMIC_SCALES_STRIDE_0}

#ifdef SEISMIC_FORMING_READOUT_PLANES_ROWS_GEMV
template <uint ROWS, uint LANES>
kernel void readout_planes_rows_gemv(PLANES_ROWS_ARGUMENTS,
    threadgroup uchar *shared [[threadgroup(0)]],
    uint tile [[threadgroup_position_in_grid]],
    uint simdgroups [[simdgroups_per_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    PLANES_ROWS_OPERANDS;
    uint rows = uint(SEISMIC_DIM_O);
    PROJECTION_SQUARES_SHARED(squares, decltype(in)::parts);
    projection::threadgroup_squares_runtime(in, rows, squares, simdgroups, sg, lane);
    const auto x = projection::shared_norm(in, squares);
    // The certified survivor pass and its full small-row fallback must use
    // the same dot recipe, so a near tie cannot change selection merely
    // because a candidate survived pruning. Keep the existing output tile
    // geometry, but each simdgroup sums every row with all 32 lanes.
    const progressive::Planes planes{top, SEISMIC_TOP_STRIDE_0, bit3,
        SEISMIC_BIT3_STRIDE_0, rest, SEISMIC_REST_STRIDE_0,
        SEISMIC_REST_STRIDE_1, SEISMIC_REST_STRIDE_2,
        scales, SEISMIC_SCALES_STRIDE_0};
    // The rows are normalized once per threadgroup into `shared`, in A
    // storage (the values `x.load8` returns), quarter-major per row
    // ([row][quarter][group]) so a simdgroup's reads are contiguous, in
    // chunks of whole lane strides that fit the declared GEMV staging
    // (`gemv_stage_packet_rows` packet rows of four words).
    // At most PROGRESSIVE_ROWS rows (the launch's `O <= 8`), so a chunk is
    // at least one lane stride.
    static_assert(projection::gemv_stage_packet_rows / PROGRESSIVE_ROWS >= 32u,
        "the staging holds a lane stride of every row");
    const uint groups = k / 32u;
    const uint capacity = projection::gemv_stage_packet_rows;
    const uint chunk = rows * groups <= capacity ? groups : capacity / rows / 32u * 32u;
    threadgroup uint4 *stage = reinterpret_cast<threadgroup uint4 *>(shared);
    const uint tid = sg * 32u + lane, threads = simdgroups * 32u;
    constexpr uint outputs = ROWS * 32u / LANES;
    const uint first = (tile * simdgroups + sg) * outputs;
    float acc[outputs][PROGRESSIVE_ROWS] = {};
    for (uint start = 0; start < groups; start += chunk) {
        const uint count = min(chunk, groups - start);
        // The squares (first chunk) or the previous chunk's reads finish
        // before this chunk is staged.
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint item = tid; item < rows * count * 4u; item += threads) {
            const uint m = item / (count * 4u), w = item % (count * 4u);
            const uint g = w / 4u, q = w % 4u;
            float4 even, odd;
            x.load8(m, (start + g) * 32u + q * 8u, 0.0f, even, odd);
            stage[(m * 4u + q) * count + g] = activation::pack8(even, odd);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        PROJECTION_UNROLL
        for (uint r = 0; r < outputs; ++r) {
            const uint n = first + r;
            if (n >= uint(SEISMIC_DIM_V))
                break;
            for (uint g = lane; g < count; g += 32u) {
                const auto codes = progressive::exact_codes(planes.group(n, start + g));
                const float scale = planes.scale(n, start + g);
                PROJECTION_UNROLL
                for (uint m = 0; m < PROGRESSIVE_ROWS; ++m) {
                    if (m >= rows) break;
                    float4 xe[4], xo[4];
                    PROJECTION_UNROLL
                    for (uint q = 0; q < 4u; ++q)
                        activation::split8(stage[(m * 4u + q) * count + g], xe[q], xo[q]);
                    acc[r][m] += progressive::exact(scale, codes, xe, xo);
                }
            }
        }
    }
    PROJECTION_UNROLL
    for (uint r = 0; r < outputs; ++r) {
        const uint n = first + r;
        if (n >= uint(SEISMIC_DIM_V))
            break;
        PROJECTION_UNROLL
        for (uint m = 0; m < PROGRESSIVE_ROWS; ++m) {
            if (m >= rows) break;
            const float logit = simd_sum(acc[r][m]);
            if (lane == 0u)
                out.store(m, n, logit);
        }
    }
}
#endif

#ifdef SEISMIC_FORMING_READOUT_PLANES_ROWS_BATCH
template <uint BATCH_ROWS>
kernel void readout_planes_rows_batch(PLANES_ROWS_ARGUMENTS,
    threadgroup uchar *shared [[threadgroup(0)]],
    uint tile [[threadgroup_position_in_grid]],
    uint simdgroups [[simdgroups_per_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    PLANES_ROWS_OPERANDS;
    PROJECTION_SQUARES_SHARED(squares, decltype(in)::parts);
    projection::threadgroup_squares_runtime(in, uint(SEISMIC_DIM_O), squares, simdgroups, sg, lane);
    const auto x = projection::shared_norm(in, squares);
    projection::gemv_batch_runtime<packets::ProgressiveExact, BATCH_ROWS>(x, out, w,
        uint(SEISMIC_DIM_O), uint(SEISMIC_DIM_V), k, tile, shared, simdgroups, sg, lane);
}
#endif

#ifdef SEISMIC_FORMING_READOUT_PLANES_ROWS_STAGE
kernel void readout_planes_rows_stage(PLANES_ROWS_ARGUMENTS,
    uint item [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]]) {
    PROJECTION_NORMALIZE_SHARED(norms);
    PLANES_ROWS_OPERANDS;
    projection::device_normalize<256>(in, item, normalized, k, norms, thread_index);
}
#endif

#ifdef SEISMIC_FORMING_READOUT_PLANES_ROWS_GEMM
template <uint TILE_M, uint TILE_N>
kernel void readout_planes_rows_gemm(PLANES_ROWS_ARGUMENTS,
    uint2 tile [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    PROJECTION_GEMM_SHARED(shared, TILE_M, TILE_N);
    PLANES_ROWS_OPERANDS;
    projection::Plain<activation, projection::AllRows> x{normalized, k, 1, k, {}};
    projection::gemm<packets::ProgressiveExact, TILE_M, TILE_N>(x, out, w, uint(SEISMIC_DIM_O),
        uint(SEISMIC_DIM_V), k, tile.y, tile.x, shared, sg, lane);
}
#endif
