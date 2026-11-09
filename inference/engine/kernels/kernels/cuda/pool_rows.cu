// pool_rows on CUDA (contract and portable body in pooled.seismic), the
// Metal launches (`metal/pool_rows.metal`):
//
// append:   a warp per row; its u (then g) and coordinates to the window
//           planes.
// complete: a warp per row, lane l holding columns [l E, (l + 1) E); a
//           completing row gathers its group (the last RATE - 1 rows of its
//           window spans, then its own inputs), pools it, RMS-norms it with
//           `norm` (the latent, published), rotates it at the coordinates
//           stored with the group's first token (`rotary::layout`), rounds it
//           (`rounding::round_lanes`; a rounded row held in A after its
//           pooling, norm and rotation) and stores it in the block plane,
//           publishing the latent, its block row and the origin coordinates.
//           Other rows publish a zero latent, block row -1 and zero
//           coordinates.
//
// Dynamic extents, strides and parameters read `seismic_words`.

#include "lib/core/activation.cuh"
#include "lib/core/head.cuh"
#include "lib/core/rotary.cuh"
#include "lib/core/rounding.cuh"
#include "lib/indexer/indexer.cuh"
#include <seismic/slab.cuh>

namespace pooled {

typedef unsigned int u32;
typedef unsigned long long u64;
typedef element::Act Act;

constexpr int W = 2 * SEISMIC_DIM_P + SEISMIC_DIM_S;
constexpr int E = W / 32;
constexpr int RATE = SEISMIC_DIM_RATE;
constexpr bool GATED = SEISMIC_DIM_GT != 0;
// A window row: u, then (gated) g.
constexpr int WIDTH = (GATED ? 2 : 1) * W;
static_assert(W % 32 == 0, "a pooled row must cover whole lanes");
static_assert(RATE >= 2 && RATE <= 16, "a group holds 2 to 16 tokens");

// A pooled row's rotary pairs: the reading attention's key layout (RA, RT).
typedef ROTARY_ENTRY_LAYOUT(W) row_layout;

} // namespace pooled

extern "C" __global__ void pool_rows_append(SEISMIC_KERNEL_PARAMS) {
    using namespace pooled;
    const u64 row = blockIdx.x;
    const int lane = threadIdx.x;
    const int destination = reinterpret_cast<const int *>(SEISMIC_PTR(SEISMIC_BUFFER_WINDOW_DESTINATIONS))
        [row * SEISMIC_WINDOW_DESTINATIONS_STRIDE_0];
    if (destination < 0) return;
    const u64 slab_rows = SEISMIC_PARAM_WINDOW_SLAB_ROWS;
    float *target = reinterpret_cast<float *>(
        slab::row(reinterpret_cast<const u32 *>(SEISMIC_PTR(SEISMIC_BUFFER_WINDOW_ROWS)),
                  static_cast<u64>(destination), slab_rows, WIDTH * sizeof(float)));
    const float *u = reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_U));
    for (int column = lane; column < W; column += 32)
        target[column] = u[row * SEISMIC_U_STRIDE_0 + column * SEISMIC_U_STRIDE_1];
    if constexpr (GATED) {
        const float *g = reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_G));
        for (int column = lane; column < W; column += 32)
            target[W + column] = g[row * SEISMIC_G_STRIDE_0 + column * SEISMIC_G_STRIDE_1];
    }
    if (lane < 4) {
        const int *coordinates = reinterpret_cast<const int *>(SEISMIC_PTR(SEISMIC_BUFFER_COORDINATES));
        reinterpret_cast<int *>(slab::row(reinterpret_cast<const u32 *>(SEISMIC_PTR(SEISMIC_BUFFER_WINDOW_COORDINATES)),
                                          static_cast<u64>(destination), slab_rows, 4 * sizeof(int)))[lane] =
            coordinates[row * SEISMIC_COORDINATES_STRIDE_0 + lane * SEISMIC_COORDINATES_STRIDE_1];
    }
}

extern "C" __global__ void pool_rows_complete(SEISMIC_KERNEL_PARAMS) {
    using namespace pooled;
    extern __shared__ float exchange[];  // [W]: the rotation's partner exchange
    const u64 row = blockIdx.x;
    const int lane = threadIdx.x;
    const int destination = reinterpret_cast<const int *>(SEISMIC_PTR(SEISMIC_BUFFER_BLOCK_DESTINATIONS))
        [row * SEISMIC_BLOCK_DESTINATIONS_STRIDE_0];
    float *published = reinterpret_cast<float *>(SEISMIC_PTR(SEISMIC_RESULT_0_BUFFER)) + row * SEISMIC_RESULT_0_STRIDE_0;
    int *latent_rows = reinterpret_cast<int *>(SEISMIC_PTR(SEISMIC_RESULT_1_BUFFER));
    int *latent_coordinates =
        reinterpret_cast<int *>(SEISMIC_PTR(SEISMIC_RESULT_2_BUFFER)) + row * SEISMIC_RESULT_2_STRIDE_0;
    if (destination < 0) {
#pragma unroll
        for (int i = 0; i < E; ++i) published[static_cast<u64>(lane * E + i) * SEISMIC_RESULT_0_STRIDE_1] = 0.0f;
        if (lane == 0) latent_rows[row * SEISMIC_RESULT_1_STRIDE_0] = -1;
        if (lane < 4) latent_coordinates[lane * SEISMIC_RESULT_2_STRIDE_1] = 0;
        return;
    }
    const u64 window_slab_rows = SEISMIC_PARAM_WINDOW_SLAB_ROWS;
    const u32 *window_rows = reinterpret_cast<const u32 *>(SEISMIC_PTR(SEISMIC_BUFFER_WINDOW_ROWS));
    // The group's earlier tokens: the last RATE - 1 rows of the window spans.
    const u64 spans = SEISMIC_DIM_R;
    const int *row_spans =
        reinterpret_cast<const int *>(SEISMIC_PTR(SEISMIC_BUFFER_WINDOW_VISIBLE)) + row * SEISMIC_WINDOW_VISIBLE_STRIDE_0;
    const u64 span_stride = SEISMIC_WINDOW_VISIBLE_STRIDE_1;
    const u64 bound_stride = SEISMIC_WINDOW_VISIBLE_STRIDE_2;
    u32 earlier = 0;
    for (u64 span = 0; span < spans; ++span)
        earlier += indexer::span_length(row_spans[span * span_stride], row_spans[span * span_stride + bound_stride]);
    // A completing row's window spans hold its group's earlier tokens.
    if (earlier < static_cast<u32>(RATE - 1)) __trap();
    const u32 first_member = earlier - static_cast<u32>(RATE - 1);
    // Member m's u, and (gated) its gate scores: a window row's follow its u,
    // the row's own are g.
    const float *u = reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_U));
    const float *g = reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_G));
    const float *members[RATE];
#pragma unroll
    for (int member = 0; member + 1 < RATE; ++member)
        members[member] = reinterpret_cast<const float *>(
            slab::row(window_rows,
                      static_cast<u64>(
                          indexer::span_row(row_spans, spans, span_stride, bound_stride, first_member + member)),
                      window_slab_rows, WIDTH * sizeof(float)));
    auto value = [&](int member, int column) {
        return member + 1 < RATE ? members[member][column]
                                 : u[row * SEISMIC_U_STRIDE_0 + static_cast<u64>(column) * SEISMIC_U_STRIDE_1];
    };
    float x[E];
    if constexpr (GATED) {
        auto gate = [&](int member, int column) {
            return member + 1 < RATE ? members[member][W + column]
                                     : g[row * SEISMIC_G_STRIDE_0 + static_cast<u64>(column) * SEISMIC_G_STRIDE_1];
        };
        float top[E], total[E];
#pragma unroll
        for (int i = 0; i < E; ++i) {
            top[i] = -__int_as_float(0x7f800000);
            total[i] = 0.0f;
            x[i] = 0.0f;
        }
        for (int member = 0; member < RATE; ++member)
#pragma unroll
            for (int i = 0; i < E; ++i) top[i] = fmaxf(top[i], gate(member, lane * E + i));
        for (int member = 0; member < RATE; ++member)
#pragma unroll
            for (int i = 0; i < E; ++i) total[i] += expf(gate(member, lane * E + i) - top[i]);
        for (int member = 0; member < RATE; ++member)
#pragma unroll
            for (int i = 0; i < E; ++i)
                x[i] += expf(gate(member, lane * E + i) - top[i]) / total[i] * value(member, lane * E + i);
    } else {
#pragma unroll
        for (int i = 0; i < E; ++i) x[i] = 0.0f;
        for (int member = 0; member < RATE; ++member)
#pragma unroll
            for (int i = 0; i < E; ++i) x[i] += value(member, lane * E + i);
#pragma unroll
        for (int i = 0; i < E; ++i) x[i] /= static_cast<float>(RATE);
    }
    // A rounded row is rounded from values held in A: the pooled row, the
    // latent and the rotated row (`pooled.seismic`).
    const rounding::Rounding trained = rounding::of(SEISMIC_PARAM_BLOCK, SEISMIC_PARAM_ELEMENT, SEISMIC_PARAM_SCALE,
                                                   SEISMIC_PARAM_AMAX_FLOOR);
    rounding::hold_lanes<Act>(x, trained);
    // The latent: RMS-normed with `norm`.
    head::rms_norm(x, reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_NORM)), SEISMIC_NORM_STRIDE_1,
                   element::word_f32(SEISMIC_PARAM_EPSILON), lane);
    rounding::hold_lanes<Act>(x, trained);
#pragma unroll
    for (int i = 0; i < E; ++i) published[static_cast<u64>(lane * E + i) * SEISMIC_RESULT_0_STRIDE_1] = x[i];
    if (lane == 0) latent_rows[row * SEISMIC_RESULT_1_STRIDE_0] = destination;
    // Rotated at the group's first token, rounded, stored.
    const int *origin = reinterpret_cast<const int *>(
        slab::row(reinterpret_cast<const u32 *>(SEISMIC_PTR(SEISMIC_BUFFER_WINDOW_COORDINATES)),
                  static_cast<u64>(indexer::span_row(row_spans, spans, span_stride, bound_stride, first_member)),
                  window_slab_rows, 4 * sizeof(int)));
    if (lane < 4) latent_coordinates[lane * SEISMIC_RESULT_2_STRIDE_1] = origin[lane];
    const rotary::table table = ROTARY_ENTRY_TABLE(origin, 1);
    row_layout::rotate(x, table, exchange, lane);
    rounding::hold_lanes<Act>(x, trained);
    rounding::round_lanes<E>(x, trained);
    // Published in the activation element, which holds every rounded value.
    element::u8 *target = slab::row(reinterpret_cast<const u32 *>(SEISMIC_PTR(SEISMIC_BUFFER_BLOCK_ROWS)),
                                    static_cast<u64>(destination), SEISMIC_PARAM_BLOCK_SLAB_ROWS, W * Act::bytes);
#pragma unroll
    for (int i = 0; i < E; ++i) element::put<Act>(target, static_cast<u64>(lane * E + i), x[i]);
}
