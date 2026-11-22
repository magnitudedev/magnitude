// pool_rows on CUDA (contract and portable body in pooled.seismic), the
// Metal launches (`metal/pool_rows.metal`):
//
// append:   a warp per row, lane l holding columns [l E, (l + 1) E); its u
//           (LayerNorm-ed when TN = 1, then g) and coordinates to the window
//           planes.
// complete: a warp per row, lane l holding columns [l E, (l + 1) E); a
//           completing row gathers its group (the last RATE - 1 rows of its
//           window spans, located cooperatively via `reduce::span_total` and
//           `reduce::span_locate`, then its own inputs, normed as append's;
//           its gates plus their place terms), pools it, RMS-norms it with
//           `norm` when TN = 0 (the latent, published), rotates it at the
//           coordinates stored with the group's first token
//           (`rotary::layout`), rounds it (`rounding::round_lanes`; a
//           rounded row held in A after its pooling, norm and rotation) and
//           stores it in the block plane, publishing the latent, its block
//           row and the origin coordinates. Other rows publish a zero latent,
//           block row -1 and zero coordinates.
//
// Dynamic extents, strides and parameters read `seismic_words`.

#define KERNEL_A SEISMIC_ELEMENT_A
#include "lib/core/activation.cuh"
#include "lib/core/head.cuh"
#include "lib/core/norm.cuh"
#include "lib/core/reduce.cuh"
#include "lib/core/rotary.cuh"
#include "lib/core/rounding.cuh"
#include "lib/indexer/indexer.cuh"
#include <seismic/slab.cuh>

namespace pooled {

typedef unsigned int u32;
typedef unsigned long long u64;
typedef activation::Act Act;
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

// Loads lane's contiguous columns [col_base, col_base + E) from `base`
// (stepping by `stride`), using 128-bit float4 loads when stride == 1 and
// 16-byte aligned, with scalar tail.
__device__ __forceinline__ void load_f32_lane(float (&out)[E], const float *base, u64 stride, u32 col_base) {
    if constexpr (E >= 4) {
        if (stride == 1 && (reinterpret_cast<u64>(base + col_base) & 15u) == 0u) {
            constexpr int VEC = E & ~3;
#pragma unroll
            for (int k = 0; k < VEC; k += 4) {
                const float4 v = *reinterpret_cast<const float4 *>(base + col_base + static_cast<u32>(k));
                out[k] = v.x;
                out[k + 1] = v.y;
                out[k + 2] = v.z;
                out[k + 3] = v.w;
            }
#pragma unroll
            for (int k = VEC; k < E; ++k) out[k] = base[col_base + static_cast<u32>(k)];
            return;
        }
    }
#pragma unroll
    for (int k = 0; k < E; ++k) out[k] = base[static_cast<u64>(col_base + static_cast<u32>(k)) * stride];
}

// Stores lane's contiguous columns [col_base, col_base + E) into `out`
// (stepping by `stride`), using 128-bit float4 stores when stride == 1 and
// 16-byte aligned, with scalar tail.
__device__ __forceinline__ void store_f32_lane(const element::Out<element::F32> &out, u32 col_base, u64 stride,
                                               const float (&values)[E]) {
    if constexpr (E >= 4) {
        if (stride == 1 && (reinterpret_cast<u64>(out.stored(col_base)) & 15u) == 0u) {
            constexpr int VEC = E & ~3;
#pragma unroll
            for (int k = 0; k < VEC; k += 4) {
                const float chunk[4] = {values[k], values[k + 1], values[k + 2], values[k + 3]};
                out.put_span<4>(col_base + static_cast<u32>(k), chunk);
            }
#pragma unroll
            for (int k = VEC; k < E; ++k) out.put(col_base + static_cast<u32>(k), values[k]);
            return;
        }
    }
#pragma unroll
    for (int k = 0; k < E; ++k) out.put(static_cast<u64>(col_base + static_cast<u32>(k)) * stride, values[k]);
}

// A row's own u (`u` at its row, stepping by `u_stride`), the warp's lane
// holding contiguous columns [lane * E, (lane + 1) * E), LayerNorm-ed per
// token with `weight` and `bias` when TN = 1 (`token_norm`): `row_norm`'s
// arithmetic over the warp's partials.
__device__ __forceinline__ void own_row(float (&x)[E], const float *u, u64 u_stride, const float *weight,
                                        u64 weight_stride, const float *bias, u64 bias_stride, float epsilon,
                                        u32 lane) {
    const u32 base = lane * static_cast<u32>(E);
    load_f32_lane(x, u, u_stride, base);
    if constexpr (TOKEN_NORM) {
        constexpr int VEC = E & ~3;
        float sum = 0.0f;
#pragma unroll
        for (int k = 0; k < E; ++k) sum += x[k];
        const float mean = row_norm::mean(seismic_warp_sum_f32(sum), W);
        float4 sq4 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
#pragma unroll
        for (int k = 0; k < VEC; k += 4) {
            x[k] -= mean;
            x[k + 1] -= mean;
            x[k + 2] -= mean;
            x[k + 3] -= mean;
            sq4 = row_norm::square4(make_float4(x[k], x[k + 1], x[k + 2], x[k + 3]), sq4);
        }
        float squares = row_norm::sum4(sq4);
#pragma unroll
        for (int k = VEC; k < E; ++k) {
            x[k] -= mean;
            squares = row_norm::square(x[k], squares);
        }
        const float inverse = row_norm::inverse(seismic_warp_sum_f32(squares), W, epsilon);
        float w[E], b[E];
        load_f32_lane(w, weight, weight_stride, base);
        load_f32_lane(b, bias, bias_stride, base);
#pragma unroll
        for (int k = 0; k < VEC; k += 4) {
            const float4 y4 = row_norm::apply4<true, true>(make_float4(x[k], x[k + 1], x[k + 2], x[k + 3]), inverse,
                                                           make_float4(w[k], w[k + 1], w[k + 2], w[k + 3]),
                                                           make_float4(b[k], b[k + 1], b[k + 2], b[k + 3]));
            x[k] = y4.x;
            x[k + 1] = y4.y;
            x[k + 2] = y4.z;
            x[k + 3] = y4.w;
        }
#pragma unroll
        for (int k = VEC; k < E; ++k) x[k] = row_norm::apply<true, true>(x[k], inverse, w[k], b[k]);
    }
}

} // namespace pooled

extern "C" __global__ void pool_rows_append(SEISMIC_KERNEL_PARAMS) {
    using namespace pooled;
    const u64 row = blockIdx.x;
    const u32 lane = threadIdx.x;
    const int destination = reinterpret_cast<const int *>(SEISMIC_PTR(SEISMIC_BUFFER_WINDOW_DESTINATIONS))
        [row * SEISMIC_WINDOW_DESTINATIONS_STRIDE_0];
    if (destination < 0) return;
    const u64 slab_rows = SEISMIC_PARAM_WINDOW_SLAB_ROWS;
    const element::Out<element::F32> target =
        SEISMIC_OUT_WINDOW_ROWS.row(static_cast<u64>(destination), slab_rows, WIDTH);
    const u32 lane_base = lane * static_cast<u32>(E);
    float x[E];
    own_row(x, reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_U)) + row * SEISMIC_U_STRIDE_0,
            SEISMIC_U_STRIDE_1, reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_NORM)),
            SEISMIC_NORM_STRIDE_1, reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_BIAS)),
            SEISMIC_BIAS_STRIDE_1, element::word_f32(SEISMIC_PARAM_EPSILON), lane);
    store_f32_lane(target, lane_base, 1, x);
    if constexpr (GATED) {
        const float *g_row = reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_G)) + row * SEISMIC_G_STRIDE_0;
        float g_vals[E];
        load_f32_lane(g_vals, g_row, SEISMIC_G_STRIDE_1, lane_base);
        store_f32_lane(target, static_cast<u32>(W) + lane_base, 1, g_vals);
    }
    if (lane < 4u) {
        const int *coordinates = reinterpret_cast<const int *>(SEISMIC_PTR(SEISMIC_BUFFER_COORDINATES));
        reinterpret_cast<int *>(slab::row(reinterpret_cast<const u32 *>(SEISMIC_PTR(SEISMIC_BUFFER_WINDOW_COORDINATES)),
                                          static_cast<u64>(destination), slab_rows, 4 * sizeof(int)))[lane] =
            coordinates[row * SEISMIC_COORDINATES_STRIDE_0 + static_cast<u64>(lane) * SEISMIC_COORDINATES_STRIDE_1];
    }
}

extern "C" __global__ void pool_rows_complete(SEISMIC_KERNEL_PARAMS) {
    using namespace pooled;
    extern __shared__ float exchange[];  // [W]: the rotation's partner exchange
    const u64 row = blockIdx.x;
    const u32 lane = threadIdx.x;
    const u32 lane_base = lane * static_cast<u32>(E);
    const int destination = reinterpret_cast<const int *>(SEISMIC_PTR(SEISMIC_BUFFER_BLOCK_DESTINATIONS))
        [row * SEISMIC_BLOCK_DESTINATIONS_STRIDE_0];
    const element::Out<element::F32> published = SEISMIC_RESULT_0_OUT.offset(row * SEISMIC_RESULT_0_STRIDE_0);
    int *latent_rows = reinterpret_cast<int *>(SEISMIC_PTR(SEISMIC_RESULT_1_BUFFER));
    int *latent_coordinates =
        reinterpret_cast<int *>(SEISMIC_PTR(SEISMIC_RESULT_2_BUFFER)) + row * SEISMIC_RESULT_2_STRIDE_0;
    if (destination < 0) {
        float zeros[E];
#pragma unroll
        for (int i = 0; i < E; ++i) zeros[i] = 0.0f;
        store_f32_lane(published, lane_base, SEISMIC_RESULT_0_STRIDE_1, zeros);
        if (lane == 0u) latent_rows[row * SEISMIC_RESULT_1_STRIDE_0] = -1;
        if (lane < 4u) latent_coordinates[static_cast<u64>(lane) * SEISMIC_RESULT_2_STRIDE_1] = 0;
        return;
    }
    const u64 window_slab_rows = SEISMIC_PARAM_WINDOW_SLAB_ROWS;
    const element::SlabOut<element::F32> window_rows = SEISMIC_OUT_WINDOW_ROWS;
    // The group's earlier tokens: the last RATE - 1 rows of the window spans,
    // located cooperatively across the warp.
    const u32 spans = static_cast<u32>(SEISMIC_DIM_R);
    const int *row_spans =
        reinterpret_cast<const int *>(SEISMIC_PTR(SEISMIC_BUFFER_WINDOW_VISIBLE)) + row * SEISMIC_WINDOW_VISIBLE_STRIDE_0;
    const u64 span_stride = SEISMIC_WINDOW_VISIBLE_STRIDE_1;
    const u64 bound_stride = SEISMIC_WINDOW_VISIBLE_STRIDE_2;
    auto span_len = [&](u64 s) {
        return indexer::span_length(row_spans[s * span_stride], row_spans[s * span_stride + bound_stride]);
    };
    const u32 earlier = static_cast<u32>(reduce::span_total(spans, span_len, static_cast<int>(lane)));
    // A completing row's window spans hold its group's earlier tokens.
    if (earlier < static_cast<u32>(RATE - 1)) __trap();
    const u32 first_member = earlier - static_cast<u32>(RATE - 1);
    const reduce::Located start =
        reduce::span_locate(spans, static_cast<long long>(first_member), span_len, static_cast<int>(lane));
    // Member m's u, and (gated) its gate scores: a window row's follow its u
    // (normed when TN = 1), the row's own are g; its own u is normed here.
    const float *g_row = reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_G)) + row * SEISMIC_G_STRIDE_0;
    const float *position = reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_POSITION));
    float own[E];
    own_row(own, reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_U)) + row * SEISMIC_U_STRIDE_0,
            SEISMIC_U_STRIDE_1, reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_NORM)),
            SEISMIC_NORM_STRIDE_1, reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_BIAS)),
            SEISMIC_BIAS_STRIDE_1, element::word_f32(SEISMIC_PARAM_EPSILON), lane);
    const float *members[RATE];
    int first_row = 0;
    u64 span = start.span;
    u32 offset = static_cast<u32>(start.offset);
    for (int member = 0; member + 1 < RATE; ++member) {
        const u32 ord = first_member + static_cast<u32>(member);
        for (;;) {
            const u32 len = span_len(span);
            if (ord < offset + len) break;
            offset += len;
            ++span;
        }
        const int at = row_spans[span * span_stride] + static_cast<int>(ord - offset);
        if (member == 0) first_row = at;
        members[member] = reinterpret_cast<const float *>(
            window_rows.view_row(static_cast<u64>(at), window_slab_rows, WIDTH).bytes());
    }
    float x[E];
    if constexpr (GATED) {
        float top[E], total[E];
#pragma unroll
        for (int i = 0; i < E; ++i) {
            top[i] = -__int_as_float(0x7f800000);
            total[i] = 0.0f;
            x[i] = 0.0f;
        }
        float scores[RATE][E];
#pragma unroll
        for (int member = 0; member < RATE; ++member) {
            if (member + 1 < RATE) {
                load_f32_lane(scores[member], members[member] + static_cast<u32>(W), 1, lane_base);
            } else {
                load_f32_lane(scores[member], g_row, SEISMIC_G_STRIDE_1, lane_base);
            }
            if (member < PLACES) {
                float place[E];
                load_f32_lane(place, position + static_cast<u64>(member) * SEISMIC_POSITION_STRIDE_0,
                              SEISMIC_POSITION_STRIDE_1, lane_base);
#pragma unroll
                for (int i = 0; i < E; ++i) scores[member][i] += place[i];
            }
#pragma unroll
            for (int i = 0; i < E; ++i) top[i] = fmaxf(top[i], scores[member][i]);
        }
#pragma unroll
        for (int member = 0; member < RATE; ++member) {
#pragma unroll
            for (int i = 0; i < E; ++i) {
                const float w = expf(scores[member][i] - top[i]);
                scores[member][i] = w;
                total[i] += w;
            }
        }
#pragma unroll
        for (int member = 0; member < RATE; ++member) {
            float val[E];
            if (member + 1 < RATE) {
                load_f32_lane(val, members[member], 1, lane_base);
            } else {
#pragma unroll
                for (int i = 0; i < E; ++i) val[i] = own[i];
            }
#pragma unroll
            for (int i = 0; i < E; ++i) x[i] += scores[member][i] / total[i] * val[i];
        }
    } else {
#pragma unroll
        for (int i = 0; i < E; ++i) x[i] = 0.0f;
#pragma unroll
        for (int member = 0; member < RATE; ++member) {
            if (member + 1 < RATE) {
                float val[E];
                load_f32_lane(val, members[member], 1, lane_base);
#pragma unroll
                for (int i = 0; i < E; ++i) x[i] += val[i];
            } else {
#pragma unroll
                for (int i = 0; i < E; ++i) x[i] += own[i];
            }
        }
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
                       element::word_f32(SEISMIC_PARAM_EPSILON), static_cast<int>(lane));
        rounding::hold_lanes<Act>(x, trained);
    }
    store_f32_lane(published, lane_base, SEISMIC_RESULT_0_STRIDE_1, x);
    if (lane == 0u) latent_rows[row * SEISMIC_RESULT_1_STRIDE_0] = destination;
    // Rotated at the group's first token, rounded, stored.
    const int *origin = reinterpret_cast<const int *>(
        slab::row(reinterpret_cast<const u32 *>(SEISMIC_PTR(SEISMIC_BUFFER_WINDOW_COORDINATES)),
                  static_cast<u64>(first_row), window_slab_rows, 4 * sizeof(int)));
    if (lane < 4u) latent_coordinates[static_cast<u64>(lane) * SEISMIC_RESULT_2_STRIDE_1] = origin[lane];
    const rotary::table table = ROTARY_ENTRY_TABLE(origin, 1);
    row_layout::rotate(x, table, exchange, static_cast<int>(lane));
    rounding::hold_lanes<Act>(x, trained);
    rounding::round_lanes<E>(x, trained);
    // Stored in the row element Y (A holds every rounded value; F32 keys hold every value).
    const element::Out<Row> target =
        SEISMIC_OUT_BLOCK_ROWS.row(static_cast<u64>(destination), SEISMIC_PARAM_BLOCK_SLAB_ROWS, W);
    if constexpr (E % 4 == 0) {
        if (Row::bytes == 2 || (reinterpret_cast<u64>(target.stored(lane_base)) & 15u) == 0u) {
            target.put_span<E>(lane_base, x);
            return;
        }
    }
#pragma unroll
    for (int i = 0; i < E; ++i) target.put(lane_base + static_cast<u32>(i), x[i]);
}
