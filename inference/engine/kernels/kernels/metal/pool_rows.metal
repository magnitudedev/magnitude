// pool_rows on Metal (contract and portable body in pooled.seismic).
//
// append:   threadgroup per row; its u (then g) and coordinates to the
//           window planes.
// complete: one simdgroup per row, lane l holding columns [l E, (l + 1) E);
//           a completing row gathers its group (the last RATE - 1 rows of
//           its window spans, then its own inputs), pools it, RMS-norms it
//           with `norm` (the latent, published), rotates it at the
//           coordinates stored with the group's first token
//           (`rotary::layout`), rounds it (`rounding::round_lanes`; a
//           rounded row held in A after its pooling, norm and rotation) and
//           stores it in the block plane, publishing the latent, its block
//           row and the origin coordinates. Other rows publish a zero latent,
//           block row -1 and zero coordinates.

#include "lib/core/activation.h"
#include "lib/core/head.h"
#include "lib/core/rotary.h"
#include "lib/core/rounding.h"
#include "lib/indexer/indexer.h"
#include <seismic/slab.h>

#define POOL_W uint(2 * SEISMIC_DIM_P + SEISMIC_DIM_S)
#define POOL_E (POOL_W / 32)
#define POOL_RATE uint(SEISMIC_DIM_RATE)
#define POOL_GATED (SEISMIC_DIM_GT != 0)
#define POOL_UNROLL _Pragma("clang loop unroll(full)")

static_assert(POOL_RATE >= 2 && POOL_RATE <= 16, "a group holds 2 to 16 tokens");

namespace pooled {

// A pooled row's rotary pairs: the reading attention's key layout (RA, RT).
typedef rotary::layout<POOL_W, SEISMIC_DIM_P, (SEISMIC_DIM_RT != 0 ? SEISMIC_DIM_S : 0), (SEISMIC_DIM_RA != 0)> row_layout;

} // namespace pooled

kernel void pool_rows_append(
    device const float *u [[buffer(SEISMIC_BUFFER_U)]],
    device const float *g [[buffer(SEISMIC_BUFFER_G)]],
    device const int *coordinates [[buffer(SEISMIC_BUFFER_COORDINATES)]],
    device const int *window_destinations [[buffer(SEISMIC_BUFFER_WINDOW_DESTINATIONS)]],
    device const ulong *window_rows [[buffer(SEISMIC_BUFFER_WINDOW_ROWS)]],
    device const ulong *window_coordinates [[buffer(SEISMIC_BUFFER_WINDOW_COORDINATES)]],
    constant ulong *seismic_words [[buffer(SEISMIC_BUFFER_WORDS)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
    const ulong row = group.x;
    const int destination = window_destinations[row];
    if (destination < 0)
        return;
    const ulong slab_rows = ulong(SEISMIC_PARAM_WINDOW_SLAB_ROWS);
    constexpr uint width = (POOL_GATED ? 2 : 1) * POOL_W;
    device float *target = slab::row<float>(window_rows, ulong(destination), slab_rows, width);
    for (uint column = lane; column < POOL_W; column += 32)
        target[column] = u[row * POOL_W + column];
    if (POOL_GATED)
        for (uint column = lane; column < POOL_W; column += 32)
            target[POOL_W + column] = g[row * POOL_W + column];
    if (lane < 4)
        slab::row<int>(window_coordinates, ulong(destination), slab_rows, 4)[lane] = coordinates[row * 4 + lane];
}

kernel void pool_rows_complete(
    device const float *u [[buffer(SEISMIC_BUFFER_U)]],
    device const float *g [[buffer(SEISMIC_BUFFER_G)]],
    device const float *norm [[buffer(SEISMIC_BUFFER_NORM)]],
    device const int *rotary_components [[buffer(SEISMIC_BUFFER_ROTARY_COMPONENTS)]],
    device const float *rotary_frequencies [[buffer(SEISMIC_BUFFER_ROTARY_FREQUENCIES)]],
    device const float *rotary_amplitudes [[buffer(SEISMIC_BUFFER_ROTARY_AMPLITUDES)]],
    device const int *coordinates [[buffer(SEISMIC_BUFFER_COORDINATES)]],
    device const int *window_visible [[buffer(SEISMIC_BUFFER_WINDOW_VISIBLE)]],
    device const int *block_destinations [[buffer(SEISMIC_BUFFER_BLOCK_DESTINATIONS)]],
    device const ulong *window_rows [[buffer(SEISMIC_BUFFER_WINDOW_ROWS)]],
    device const ulong *window_coordinates [[buffer(SEISMIC_BUFFER_WINDOW_COORDINATES)]],
    device const ulong *block_rows [[buffer(SEISMIC_BUFFER_BLOCK_ROWS)]],
    device float *latent [[buffer(SEISMIC_RESULT_0_BUFFER)]],
    device int *latent_rows [[buffer(SEISMIC_RESULT_1_BUFFER)]],
    device int *latent_coordinates [[buffer(SEISMIC_RESULT_2_BUFFER)]],
    constant ulong *seismic_words [[buffer(SEISMIC_BUFFER_WORDS)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
    constexpr uint E = POOL_E;
    constexpr uint W = POOL_W;
    constexpr uint width = (POOL_GATED ? 2 : 1) * W;
    const ulong row = group.x;
    const int destination = block_destinations[row];
    device float *published = latent + row * SEISMIC_RESULT_0_STRIDE_0;
    if (destination < 0) {
        POOL_UNROLL
        for (uint i = 0; i < E; ++i)
            published[(lane * E + i) * SEISMIC_RESULT_0_STRIDE_1] = 0.0f;
        if (lane == 0)
            latent_rows[row * SEISMIC_RESULT_1_STRIDE_0] = -1;
        if (lane < 4)
            latent_coordinates[row * SEISMIC_RESULT_2_STRIDE_0 + lane * SEISMIC_RESULT_2_STRIDE_1] = 0;
        return;
    }
    const ulong window_slab_rows = ulong(SEISMIC_PARAM_WINDOW_SLAB_ROWS);
    // The group's earlier tokens: the last RATE - 1 rows of the window spans.
    const uint spans = uint(SEISMIC_DIM_R);
    device const int *row_spans = window_visible + row * spans * 2;
    uint earlier = 0;
    for (uint span = 0; span < spans; ++span)
        earlier += indexer::span_length(row_spans[span * 2], row_spans[span * 2 + 1]);
    device const float *members[POOL_RATE];
    for (uint member = 0; member + 1 < POOL_RATE; ++member) {
        const int at = indexer::span_row(row_spans, spans, earlier - (POOL_RATE - 1) + member);
        members[member] = slab::row<float>(window_rows, ulong(at), window_slab_rows, width);
    }
    members[POOL_RATE - 1] = u + row * W;
    // A member's gate scores follow its u in the window; the row's own are g.
    device const float *own_gates = g + row * W;
    float x[E];
    if (POOL_GATED) {
        float top[E], total[E];
        POOL_UNROLL
        for (uint i = 0; i < E; ++i) {
            top[i] = -INFINITY;
            total[i] = 0.0f;
            x[i] = 0.0f;
        }
        for (uint member = 0; member < POOL_RATE; ++member) {
            device const float *gates = member + 1 < POOL_RATE ? members[member] + W : own_gates;
            POOL_UNROLL
            for (uint i = 0; i < E; ++i)
                top[i] = metal::max(top[i], gates[lane * E + i]);
        }
        for (uint member = 0; member < POOL_RATE; ++member) {
            device const float *gates = member + 1 < POOL_RATE ? members[member] + W : own_gates;
            POOL_UNROLL
            for (uint i = 0; i < E; ++i)
                total[i] += metal::precise::exp(gates[lane * E + i] - top[i]);
        }
        for (uint member = 0; member < POOL_RATE; ++member) {
            device const float *gates = member + 1 < POOL_RATE ? members[member] + W : own_gates;
            POOL_UNROLL
            for (uint i = 0; i < E; ++i)
                x[i] += metal::precise::exp(gates[lane * E + i] - top[i]) / total[i] * members[member][lane * E + i];
        }
    } else {
        POOL_UNROLL
        for (uint i = 0; i < E; ++i)
            x[i] = 0.0f;
        for (uint member = 0; member < POOL_RATE; ++member) {
            POOL_UNROLL
            for (uint i = 0; i < E; ++i)
                x[i] += members[member][lane * E + i];
        }
        POOL_UNROLL
        for (uint i = 0; i < E; ++i)
            x[i] /= float(POOL_RATE);
    }
    // A rounded row is rounded from values held in A: the pooled row, the
    // latent and the rotated row (`pooled.seismic`).
    const rounding::Rounding trained = rounding::of(uint(SEISMIC_PARAM_BLOCK), uint(SEISMIC_PARAM_ELEMENT),
        uint(SEISMIC_PARAM_SCALE), uint(SEISMIC_PARAM_AMAX_FLOOR));
    rounding::hold_lanes<element::Act>(x, trained);
    // The latent: RMS-normed with `norm`.
    head::rms_norm(x, norm, as_type<float>(uint(SEISMIC_PARAM_EPSILON)), lane);
    rounding::hold_lanes<element::Act>(x, trained);
    POOL_UNROLL
    for (uint i = 0; i < E; ++i)
        published[(lane * E + i) * SEISMIC_RESULT_0_STRIDE_1] = x[i];
    if (lane == 0)
        latent_rows[row * SEISMIC_RESULT_1_STRIDE_0] = destination;
    // Rotated at the group's first token, rounded, stored.
    device const int *origin = reinterpret_cast<device const int *>(slab::row<int>(window_coordinates,
        ulong(indexer::span_row(row_spans, spans, earlier - (POOL_RATE - 1))), window_slab_rows, 4));
    if (lane < 4)
        latent_coordinates[row * SEISMIC_RESULT_2_STRIDE_0 + lane * SEISMIC_RESULT_2_STRIDE_1] = origin[lane];
    const pooled::row_layout::table table(origin, rotary_components, rotary_frequencies, rotary_amplitudes, lane);
    pooled::row_layout::rotate(table, lane, x);
    rounding::hold_lanes<element::Act>(x, trained);
    rounding::round_lanes<E>(x, trained);
    // Published in the activation element, which holds every rounded value.
    device element::Act::storage *target = reinterpret_cast<device element::Act::storage *>(
        slab::row<element::Act::native>(block_rows, ulong(destination), ulong(SEISMIC_PARAM_BLOCK_SLAB_ROWS), W));
    POOL_UNROLL
    for (uint i = 0; i < E; ++i)
        target[lane * E + i] = element::Act::store(x[i]);
}
