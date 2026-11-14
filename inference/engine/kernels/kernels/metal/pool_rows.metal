// pool_rows on Metal (contract and portable body in pooled.seismic).
//
// append:   threadgroup per row; its u (LayerNorm-ed when TN = 1, then g)
//           and coordinates to the window planes.
// complete: one simdgroup per row, lane l holding columns [l E, (l + 1) E);
//           a completing row gathers its group (the last RATE - 1 rows of
//           its window spans, then its own inputs, its gates plus their
//           place terms), pools it, RMS-norms it with `norm` when TN = 0
//           (the latent, published), rotates it at the
//           coordinates stored with the group's first token
//           (`rotary::layout`), rounds it (`rounding::round_lanes`; a
//           rounded row held in A after its pooling, norm and rotation) and
//           stores it in the block plane, publishing the latent, its block
//           row and the origin coordinates. Other rows publish a zero latent,
//           block row -1 and zero coordinates.

#include "lib/core/activation.h"
#include "lib/core/head.h"
#include "lib/core/norm.h"
#include "lib/core/rotary.h"
#include "lib/core/rounding.h"
#include "lib/indexer/indexer.h"
#include <seismic/slab.h>

#define POOL_W uint(2 * SEISMIC_DIM_P + SEISMIC_DIM_S)
#define POOL_E (POOL_W / 32)
#define POOL_RATE uint(SEISMIC_DIM_RATE)
#define POOL_GATED (SEISMIC_DIM_GT != 0)
#define POOL_TOKEN_NORM (SEISMIC_DIM_TN != 0)
#define POOL_PLACES uint(SEISMIC_DIM_PR)
#define POOL_UNROLL _Pragma("clang loop unroll(full)")

static_assert(POOL_RATE >= 2 && POOL_RATE <= 16, "a group holds 2 to 16 tokens");

namespace pooled {

// A pooled row's rotary pairs: the reading attention's key layout (RA, RT).
typedef rotary::layout<POOL_W, SEISMIC_DIM_P, (SEISMIC_DIM_RT != 0 ? SEISMIC_DIM_S : 0), (SEISMIC_DIM_RA != 0)> row_layout;
// The row element Y: A for a history source's rows, F32 for indexer keys.
typedef ELEMENT_OF(SEISMIC_ELEMENT_Y) Row;

// Lane `lane`'s k-th column: l + 32 k (`strided`, append's) or l E + k
// (complete's).
inline uint own_column(uint lane, uint k, bool strided) {
    return strided ? lane + 32 * k : lane * POOL_E + k;
}

// A row's own u, the lane holding its `own_column`s in x, LayerNorm-ed per
// token with `weight` and `bias` when TN = 1 (`token_norm`): `row_norm`'s
// arithmetic over the simdgroup's partials.
inline void own_row(thread float (&x)[POOL_E], device const float *u, device const float *weight,
                    device const float *bias, float epsilon, uint lane, bool strided) {
    POOL_UNROLL
    for (uint k = 0; k < POOL_E; ++k)
        x[k] = u[own_column(lane, k, strided)];
    if (POOL_TOKEN_NORM) {
        float sum = 0.0f;
        POOL_UNROLL
        for (uint k = 0; k < POOL_E; ++k)
            sum += x[k];
        const float mean = row_norm::mean(simd_sum(sum), POOL_W);
        float squares = 0.0f;
        POOL_UNROLL
        for (uint k = 0; k < POOL_E; ++k) {
            x[k] -= mean;
            squares = row_norm::square(x[k], squares);
        }
        const float inverse = row_norm::inverse(simd_sum(squares), POOL_W, epsilon);
        POOL_UNROLL
        for (uint k = 0; k < POOL_E; ++k) {
            const uint column = own_column(lane, k, strided);
            x[k] = row_norm::apply<true, true>(x[k], inverse, weight[column], bias[column]);
        }
    }
}

} // namespace pooled

kernel void pool_rows_append(
    device const float *u [[buffer(SEISMIC_BUFFER_U)]],
    device const float *g [[buffer(SEISMIC_BUFFER_G)]],
    device const float *norm [[buffer(SEISMIC_BUFFER_NORM)]],
    device const float *bias [[buffer(SEISMIC_BUFFER_BIAS)]],
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
    float x[POOL_E];
    pooled::own_row(x, u + row * POOL_W, norm, bias, as_type<float>(uint(SEISMIC_PARAM_EPSILON)), lane, true);
    POOL_UNROLL
    for (uint k = 0; k < POOL_E; ++k)
        target[lane + 32 * k] = x[k];
    if (POOL_GATED)
        for (uint column = lane; column < POOL_W; column += 32)
            target[POOL_W + column] = g[row * POOL_W + column];
    if (lane < 4)
        slab::row<int>(window_coordinates, ulong(destination), slab_rows, 4)[lane] = coordinates[row * 4 + lane];
}

kernel void pool_rows_complete(
    device const float *u [[buffer(SEISMIC_BUFFER_U)]],
    device const float *g [[buffer(SEISMIC_BUFFER_G)]],
    device const float *position [[buffer(SEISMIC_BUFFER_POSITION)]],
    device const float *norm [[buffer(SEISMIC_BUFFER_NORM)]],
    device const float *bias [[buffer(SEISMIC_BUFFER_BIAS)]],
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
    device const float *members[POOL_RATE - 1];
    for (uint member = 0; member + 1 < POOL_RATE; ++member) {
        const int at = indexer::span_row(row_spans, spans, earlier - (POOL_RATE - 1) + member);
        members[member] = slab::row<float>(window_rows, ulong(at), window_slab_rows, width);
    }
    const float epsilon = as_type<float>(uint(SEISMIC_PARAM_EPSILON));
    // The row's own u (normed per token when TN = 1), beside its window
    // members'.
    float own[E];
    pooled::own_row(own, u + row * W, norm, bias, epsilon, lane, false);
    // A member's gate scores follow its u in the window; the row's own are
    // g. Each takes its place's term (PR = RATE; none at PR = 0).
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
        float scores[POOL_RATE][E];
        for (uint member = 0; member < POOL_RATE; ++member) {
            device const float *gates = member + 1 < POOL_RATE ? members[member] + W : own_gates;
            POOL_UNROLL
            for (uint i = 0; i < E; ++i) {
                const float place = member < POOL_PLACES ? position[member * W + lane * E + i] : 0.0f;
                scores[member][i] = gates[lane * E + i] + place;
                top[i] = metal::max(top[i], scores[member][i]);
            }
        }
        for (uint member = 0; member < POOL_RATE; ++member) {
            POOL_UNROLL
            for (uint i = 0; i < E; ++i)
                total[i] += metal::precise::exp(scores[member][i] - top[i]);
        }
        for (uint member = 0; member < POOL_RATE; ++member) {
            POOL_UNROLL
            for (uint i = 0; i < E; ++i) {
                const float value = member + 1 < POOL_RATE ? members[member][lane * E + i] : own[i];
                x[i] += metal::precise::exp(scores[member][i] - top[i]) / total[i] * value;
            }
        }
    } else {
        POOL_UNROLL
        for (uint i = 0; i < E; ++i)
            x[i] = 0.0f;
        for (uint member = 0; member < POOL_RATE; ++member) {
            POOL_UNROLL
            for (uint i = 0; i < E; ++i)
                x[i] += member + 1 < POOL_RATE ? members[member][lane * E + i] : own[i];
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
    // The latent: RMS-normed with `norm` (TN = 0); a token-normed row's is
    // the pooled row.
    if (!POOL_TOKEN_NORM) {
        head::rms_norm(x, norm, epsilon, lane);
        rounding::hold_lanes<element::Act>(x, trained);
    }
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
    // Stored in the row element Y (A holds every rounded value; F32 keys
    // hold every value).
    device pooled::Row::storage *target = reinterpret_cast<device pooled::Row::storage *>(
        slab::row<pooled::Row::native>(block_rows, ulong(destination), ulong(SEISMIC_PARAM_BLOCK_SLAB_ROWS), W));
    POOL_UNROLL
    for (uint i = 0; i < E; ++i)
        target[lane * E + i] = pooled::Row::store(x[i]);
}
