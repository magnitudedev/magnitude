// pool_rows on Metal (contract and portable body in pooled.seismic).
//
// append:   threadgroup per row; its u (LayerNorm-ed when TN = 1, then g)
//           and coordinates to the window planes.
// complete: one simdgroup per row, lane l holding columns [l E, (l + 1) E);
//           a completing row gathers its group (the last RATE - 1 rows of
//           its window spans, located cooperatively via `reduce::span_total`
//           and `reduce::span_locate`, then its own inputs, its gates plus
//           their place terms), pools it, RMS-norms it with `norm` when
//           TN = 0 (the latent, published), rotates it at the coordinates
//           stored with the group's first token (`rotary::layout`), rounds it
//           (`rounding::round_lanes`; a rounded row held in A after its
//           pooling, norm and rotation) and stores it in the block plane,
//           publishing the latent, its block row and the origin coordinates.
//           Other rows publish a zero latent, block row -1 and zero
//           coordinates.

#define KERNEL_A SEISMIC_ELEMENT_A
#include "lib/core/activation.h"
#include "lib/core/head.h"
#include "lib/core/norm.h"
#include "lib/core/reduce.h"
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

// Loads lane's contiguous columns [col_base, col_base + POOL_E) from `base`
// using 128-bit packed_float4 loads with scalar tail.
inline void load_f32_lane(thread float (&out)[POOL_E], device const float *base, uint col_base) {
    constexpr uint VEC = POOL_E & ~3u;
    POOL_UNROLL
    for (uint k = 0; k < VEC; k += 4) {
        const float4 v = float4(*reinterpret_cast<device const packed_float4 *>(base + col_base + k));
        out[k] = v.x;
        out[k + 1] = v.y;
        out[k + 2] = v.z;
        out[k + 3] = v.w;
    }
    POOL_UNROLL
    for (uint k = VEC; k < POOL_E; ++k)
        out[k] = base[col_base + k];
}

// Stores lane's contiguous columns [col_base, col_base + POOL_E) to `target`
// using 4-wide vector stores when stride == 1, with scalar tail.
template <typename Elem>
inline void store_lane(element::Out<Elem> target, uint col_base, ulong stride, thread const float (&values)[POOL_E]) {
    if (stride == 1ul) {
        constexpr uint VEC = POOL_E & ~3u;
        POOL_UNROLL
        for (uint k = 0; k < VEC; k += 4)
            target.put4(col_base + k, float4(values[k], values[k + 1], values[k + 2], values[k + 3]));
        POOL_UNROLL
        for (uint k = VEC; k < POOL_E; ++k)
            target.put(col_base + k, values[k]);
        return;
    }
    POOL_UNROLL
    for (uint k = 0; k < POOL_E; ++k)
        target.put(ulong(col_base + k) * stride, values[k]);
}

// A row's own u, the lane holding contiguous columns [lane * E, (lane + 1) * E),
// LayerNorm-ed per token with `weight` and `bias` when TN = 1 (`token_norm`):
// `row_norm`'s arithmetic over the simdgroup's partials.
inline void own_row(thread float (&x)[POOL_E], device const float *u, device const float *weight,
                    device const float *bias, float epsilon, uint lane) {
    const uint base = lane * POOL_E;
    load_f32_lane(x, u, base);
    if constexpr (POOL_TOKEN_NORM) {
        constexpr uint VEC = POOL_E & ~3u;
        float sum = 0.0f;
        POOL_UNROLL
        for (uint k = 0; k < POOL_E; ++k)
            sum += x[k];
        const float mean = row_norm::mean(simd_sum(sum), POOL_W);
        float4 sq4 = float4(0.0f);
        POOL_UNROLL
        for (uint k = 0; k < VEC; k += 4) {
            x[k] -= mean;
            x[k + 1] -= mean;
            x[k + 2] -= mean;
            x[k + 3] -= mean;
            sq4 = row_norm::square4(float4(x[k], x[k + 1], x[k + 2], x[k + 3]), sq4);
        }
        float squares = row_norm::sum4(sq4);
        POOL_UNROLL
        for (uint k = VEC; k < POOL_E; ++k) {
            x[k] -= mean;
            squares = row_norm::square(x[k], squares);
        }
        const float inverse = row_norm::inverse(simd_sum(squares), POOL_W, epsilon);
        float w[POOL_E], b[POOL_E];
        load_f32_lane(w, weight, base);
        load_f32_lane(b, bias, base);
        POOL_UNROLL
        for (uint k = 0; k < VEC; k += 4) {
            const float4 y4 = row_norm::apply4<true, true>(
                float4(x[k], x[k + 1], x[k + 2], x[k + 3]), inverse,
                float4(w[k], w[k + 1], w[k + 2], w[k + 3]),
                float4(b[k], b[k + 1], b[k + 2], b[k + 3]));
            x[k] = y4.x;
            x[k + 1] = y4.y;
            x[k + 2] = y4.z;
            x[k + 3] = y4.w;
        }
        POOL_UNROLL
        for (uint k = VEC; k < POOL_E; ++k)
            x[k] = row_norm::apply<true, true>(x[k], inverse, w[k], b[k]);
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
    SEISMIC_OUT_WINDOW_ROWS_PARAM(window_rows),
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
    const element::Out<element::F32> target = slab::out_row(window_rows, ulong(destination), slab_rows, width);
    const uint lane_base = lane * POOL_E;
    float x[POOL_E];
    pooled::own_row(x, u + row * POOL_W, norm, bias, as_type<float>(uint(SEISMIC_PARAM_EPSILON)), lane);
    pooled::store_lane(target, lane_base, 1ul, x);
    if (POOL_GATED) {
        float g_vals[POOL_E];
        pooled::load_f32_lane(g_vals, g + row * POOL_W, lane_base);
        pooled::store_lane(target, POOL_W + lane_base, 1ul, g_vals);
    }
    if (lane < 4u)
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
    SEISMIC_OUT_WINDOW_ROWS_PARAM(window_rows),
    device const ulong *window_coordinates [[buffer(SEISMIC_BUFFER_WINDOW_COORDINATES)]],
    SEISMIC_OUT_BLOCK_ROWS_PARAM(block_rows),
    SEISMIC_RESULT_0_PARAM(latent),
    device int *latent_rows [[buffer(SEISMIC_RESULT_1_BUFFER)]],
    device int *latent_coordinates [[buffer(SEISMIC_RESULT_2_BUFFER)]],
    constant ulong *seismic_words [[buffer(SEISMIC_BUFFER_WORDS)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
    constexpr uint E = POOL_E;
    constexpr uint W = POOL_W;
    constexpr uint width = (POOL_GATED ? 2 : 1) * W;
    const ulong row = group.x;
    const uint lane_base = lane * E;
    const int destination = block_destinations[row];
    const element::Out<element::F32> published =
        element::Out<element::F32>(latent).offset(row * SEISMIC_RESULT_0_STRIDE_0);
    if (destination < 0) {
        float zeros[E];
        POOL_UNROLL
        for (uint i = 0; i < E; ++i)
            zeros[i] = 0.0f;
        pooled::store_lane(published, lane_base, SEISMIC_RESULT_0_STRIDE_1, zeros);
        if (lane == 0u)
            latent_rows[row * SEISMIC_RESULT_1_STRIDE_0] = -1;
        if (lane < 4u)
            latent_coordinates[row * SEISMIC_RESULT_2_STRIDE_0 + ulong(lane) * SEISMIC_RESULT_2_STRIDE_1] = 0;
        return;
    }
    const ulong window_slab_rows = ulong(SEISMIC_PARAM_WINDOW_SLAB_ROWS);
    // The group's earlier tokens: the last RATE - 1 rows of the window spans,
    // located cooperatively across the simdgroup.
    const uint spans = uint(SEISMIC_DIM_R);
    device const int *row_spans = window_visible + row * ulong(spans) * 2;
    auto span_len = [&](ulong s) { return indexer::span_length(row_spans[s * 2], row_spans[s * 2 + 1]); };
    const uint earlier = reduce::span_total(spans, span_len, lane);
    const uint first_member = earlier - (POOL_RATE - 1);
    const reduce::Located start = reduce::span_locate(spans, first_member, span_len, lane);
    device const float *members[POOL_RATE - 1];
    int first_row = 0;
    ulong span = start.span;
    uint offset = start.offset;
    for (uint member = 0; member + 1 < POOL_RATE; ++member) {
        const uint ord = first_member + member;
        for (;;) {
            const uint len = span_len(span);
            if (ord < offset + len)
                break;
            offset += len;
            ++span;
        }
        const int at = row_spans[span * 2] + int(ord - offset);
        if (member == 0)
            first_row = at;
        members[member] = slab::row<float>(window_rows, ulong(at), window_slab_rows, width);
    }
    const float epsilon = as_type<float>(uint(SEISMIC_PARAM_EPSILON));
    // The row's own u (normed per token when TN = 1), beside its window
    // members'.
    float own[E];
    pooled::own_row(own, u + row * W, norm, bias, epsilon, lane);
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
        POOL_UNROLL
        for (uint member = 0; member < POOL_RATE; ++member) {
            device const float *gates = member + 1 < POOL_RATE ? members[member] + W : own_gates;
            pooled::load_f32_lane(scores[member], gates, lane_base);
            if (member < POOL_PLACES) {
                float place[E];
                pooled::load_f32_lane(place, position + member * W, lane_base);
                POOL_UNROLL
                for (uint i = 0; i < E; ++i)
                    scores[member][i] += place[i];
            }
            POOL_UNROLL
            for (uint i = 0; i < E; ++i)
                top[i] = metal::max(top[i], scores[member][i]);
        }
        POOL_UNROLL
        for (uint member = 0; member < POOL_RATE; ++member) {
            POOL_UNROLL
            for (uint i = 0; i < E; ++i) {
                const float w = metal::precise::exp(scores[member][i] - top[i]);
                scores[member][i] = w;
                total[i] += w;
            }
        }
        POOL_UNROLL
        for (uint member = 0; member < POOL_RATE; ++member) {
            float val[E];
            if (member + 1 < POOL_RATE) {
                pooled::load_f32_lane(val, members[member], lane_base);
            } else {
                POOL_UNROLL
                for (uint i = 0; i < E; ++i)
                    val[i] = own[i];
            }
            POOL_UNROLL
            for (uint i = 0; i < E; ++i)
                x[i] += scores[member][i] / total[i] * val[i];
        }
    } else {
        POOL_UNROLL
        for (uint i = 0; i < E; ++i)
            x[i] = 0.0f;
        POOL_UNROLL
        for (uint member = 0; member < POOL_RATE; ++member) {
            if (member + 1 < POOL_RATE) {
                float val[E];
                pooled::load_f32_lane(val, members[member], lane_base);
                POOL_UNROLL
                for (uint i = 0; i < E; ++i)
                    x[i] += val[i];
            } else {
                POOL_UNROLL
                for (uint i = 0; i < E; ++i)
                    x[i] += own[i];
            }
        }
        POOL_UNROLL
        for (uint i = 0; i < E; ++i)
            x[i] /= float(POOL_RATE);
    }
    // A rounded row is rounded from values held in A: the pooled row, the
    // latent and the rotated row (`pooled.seismic`).
    const rounding::Rounding trained = rounding::of(uint(SEISMIC_PARAM_BLOCK), uint(SEISMIC_PARAM_ELEMENT),
        uint(SEISMIC_PARAM_SCALE), uint(SEISMIC_PARAM_AMAX_FLOOR));
    rounding::hold_lanes<activation::Act>(x, trained);
    // The latent: RMS-normed with `norm` (TN = 0); a token-normed row's is
    // the pooled row.
    if (!POOL_TOKEN_NORM) {
        head::rms_norm(x, norm, epsilon, lane);
        rounding::hold_lanes<activation::Act>(x, trained);
    }
    pooled::store_lane(published, lane_base, SEISMIC_RESULT_0_STRIDE_1, x);
    if (lane == 0u)
        latent_rows[row * SEISMIC_RESULT_1_STRIDE_0] = destination;
    // Rotated at the group's first token, rounded, stored.
    device const int *origin = reinterpret_cast<device const int *>(
        slab::row<int>(window_coordinates, ulong(first_row), window_slab_rows, 4));
    if (lane < 4u)
        latent_coordinates[row * SEISMIC_RESULT_2_STRIDE_0 + ulong(lane) * SEISMIC_RESULT_2_STRIDE_1] = origin[lane];
    const pooled::row_layout::table table(origin, rotary_components, rotary_frequencies, rotary_amplitudes, lane);
    pooled::row_layout::rotate(table, lane, x);
    rounding::hold_lanes<activation::Act>(x, trained);
    rounding::round_lanes<E>(x, trained);
    // Stored in the row element Y (A holds every rounded value; F32 keys
    // hold every value).
    const element::Out<pooled::Row> target =
        slab::out_row(block_rows, ulong(destination), ulong(SEISMIC_PARAM_BLOCK_SLAB_ROWS), W);
    pooled::store_lane(target, lane_base, 1ul, x);
}
