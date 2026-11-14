// pool_rows on CUDA (contract and portable body in pooled.seismic), the
// Metal launches (`metal/pool_rows.metal`):
//
// append:   a warp per row, lane l holding columns l + 32 k; its u
//           (LayerNorm-ed when TN = 1, then g) and coordinates to the window
//           planes.
// complete: a warp per row, lane l holding columns [l E, (l + 1) E); a
//           completing row gathers its group (the last RATE - 1 rows of its
//           window spans, then its own inputs, normed as append's; its gates
//           plus their place terms), pools it, RMS-norms it with `norm` when
//           TN = 0 (the latent, published), rotates it at the coordinates
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
#include "lib/core/norm.cuh"
#include "lib/core/rotary.cuh"
#include "lib/core/rounding.cuh"
#include "lib/indexer/indexer.cuh"
#include <seismic/slab.cuh>

namespace pooled {

typedef unsigned int u32;
typedef unsigned long long u64;
typedef element::Act Act;
// The row element Y: A for a history source's rows, F32 for indexer keys.
typedef ELEMENT_OF(SEISMIC_ELEMENT_Y) Row;

constexpr int W = 2 * SEISMIC_DIM_P + SEISMIC_DIM_S;
constexpr int E = W / 32;
constexpr int RATE = SEISMIC_DIM_RATE;
constexpr bool GATED = SEISMIC_DIM_GT != 0;
constexpr bool TOKEN_NORM = SEISMIC_DIM_TN != 0;
constexpr int PLACES = SEISMIC_DIM_PR;
// A window row: u, then (gated) g.
constexpr int WIDTH = (GATED ? 2 : 1) * W;
static_assert(W % 32 == 0, "a pooled row must cover whole lanes");
static_assert(RATE >= 2 && RATE <= 16, "a group holds 2 to 16 tokens");

// A pooled row's rotary pairs: the reading attention's key layout (RA, RT).
typedef ROTARY_ENTRY_LAYOUT(W) row_layout;

// A row's own u (`u` at its row, stepping by `u_stride`), the warp's lane
// holding column `column(k)` in x[k], LayerNorm-ed per token with `weight`
// and `bias` (each stepping by its stride) when TN = 1 (`token_norm`):
// `row_norm`'s arithmetic over the warp's partials.
template <class Column>
__device__ __forceinline__ void own_row(float (&x)[E], const float *u, u64 u_stride, const float *weight,
                                        u64 weight_stride, const float *bias, u64 bias_stride, float epsilon,
                                        Column column) {
#pragma unroll
    for (int k = 0; k < E; ++k) x[k] = u[static_cast<u64>(column(k)) * u_stride];
    if constexpr (TOKEN_NORM) {
        float sum = 0.0f;
#pragma unroll
        for (int k = 0; k < E; ++k) sum += x[k];
        const float mean = row_norm::mean(seismic_warp_sum_f32(sum), W);
        float squares = 0.0f;
#pragma unroll
        for (int k = 0; k < E; ++k) {
            x[k] -= mean;
            squares = row_norm::square(x[k], squares);
        }
        const float inverse = row_norm::inverse(seismic_warp_sum_f32(squares), W, epsilon);
#pragma unroll
        for (int k = 0; k < E; ++k) {
            const u64 c = static_cast<u64>(column(k));
            x[k] = row_norm::apply<true, true>(x[k], inverse, weight[c * weight_stride], bias[c * bias_stride]);
        }
    }
}

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
    float x[E];
    own_row(x, reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_U)) + row * SEISMIC_U_STRIDE_0,
            SEISMIC_U_STRIDE_1, reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_NORM)),
            SEISMIC_NORM_STRIDE_1, reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_BIAS)),
            SEISMIC_BIAS_STRIDE_1, element::word_f32(SEISMIC_PARAM_EPSILON), [&](int k) { return lane + 32 * k; });
#pragma unroll
    for (int k = 0; k < E; ++k) target[lane + 32 * k] = x[k];
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
    // Member m's u, and (gated) its gate scores: a window row's follow its u
    // (normed when TN = 1), the row's own are g; its own u is normed here.
    const float *g = reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_G));
    const float *position = reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_POSITION));
    float own[E];
    own_row(own, reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_U)) + row * SEISMIC_U_STRIDE_0,
            SEISMIC_U_STRIDE_1, reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_NORM)),
            SEISMIC_NORM_STRIDE_1, reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_BIAS)),
            SEISMIC_BIAS_STRIDE_1, element::word_f32(SEISMIC_PARAM_EPSILON), [&](int i) { return lane * E + i; });
    const float *members[RATE];
#pragma unroll
    for (int member = 0; member + 1 < RATE; ++member)
        members[member] = reinterpret_cast<const float *>(
            slab::row(window_rows,
                      static_cast<u64>(
                          indexer::span_row(row_spans, spans, span_stride, bound_stride, first_member + member)),
                      window_slab_rows, WIDTH * sizeof(float)));
    // Lane `lane`'s column i of member `member`.
    auto value = [&](int member, int i) { return member + 1 < RATE ? members[member][lane * E + i] : own[i]; };
    float x[E];
    if constexpr (GATED) {
        // Member `member`'s gate logit at lane `lane`'s column i, with its
        // place's term (PR = RATE; none at PR = 0).
        auto gate = [&](int member, int i) {
            const u64 column = static_cast<u64>(lane * E + i);
            const float logit = member + 1 < RATE ? members[member][W + column]
                                                  : g[row * SEISMIC_G_STRIDE_0 + column * SEISMIC_G_STRIDE_1];
            return member < PLACES
                ? logit + position[static_cast<u64>(member) * SEISMIC_POSITION_STRIDE_0 + column * SEISMIC_POSITION_STRIDE_1]
                : logit;
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
            for (int i = 0; i < E; ++i) top[i] = fmaxf(top[i], gate(member, i));
        for (int member = 0; member < RATE; ++member)
#pragma unroll
            for (int i = 0; i < E; ++i) total[i] += expf(gate(member, i) - top[i]);
        for (int member = 0; member < RATE; ++member)
#pragma unroll
            for (int i = 0; i < E; ++i)
                x[i] += expf(gate(member, i) - top[i]) / total[i] * value(member, i);
    } else {
#pragma unroll
        for (int i = 0; i < E; ++i) x[i] = 0.0f;
        for (int member = 0; member < RATE; ++member)
#pragma unroll
            for (int i = 0; i < E; ++i) x[i] += value(member, i);
#pragma unroll
        for (int i = 0; i < E; ++i) x[i] /= static_cast<float>(RATE);
    }
    // A rounded row is rounded from values held in A: the pooled row, the
    // latent and the rotated row (`pooled.seismic`).
    const rounding::Rounding trained = rounding::of(SEISMIC_PARAM_BLOCK, SEISMIC_PARAM_ELEMENT, SEISMIC_PARAM_SCALE,
                                                   SEISMIC_PARAM_AMAX_FLOOR);
    rounding::hold_lanes<Act>(x, trained);
    // The latent: RMS-normed with `norm` (TN = 0); a token-normed row's is the
    // pooled row.
    if constexpr (!TOKEN_NORM) {
        head::rms_norm(x, reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_NORM)), SEISMIC_NORM_STRIDE_1,
                       element::word_f32(SEISMIC_PARAM_EPSILON), lane);
        rounding::hold_lanes<Act>(x, trained);
    }
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
    // Stored in the row element Y (A holds every rounded value; F32 keys hold every value).
    element::u8 *target = slab::row(reinterpret_cast<const u32 *>(SEISMIC_PTR(SEISMIC_BUFFER_BLOCK_ROWS)),
                                    static_cast<u64>(destination), SEISMIC_PARAM_BLOCK_SLAB_ROWS, W * Row::bytes);
#pragma unroll
    for (int i = 0; i < E; ++i) element::put<Row>(target, static_cast<u64>(lane * E + i), x[i]);
}
