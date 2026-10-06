// Router logits of the decode rows (M <= 8), for `routed_route` and
// `routed_select`: every threadgroup reduces its rows' RMS inverses in the
// stage launch's order, then forms logit columns from the router input
// x[m, k] = round_A(r[m, k] * inverse(m) * norm[k]). A logit is a lane's
// in-order sum over its 32-value packets (lane, lane + 32, ...), then a
// simd_sum, for every column alike, so experts with equal router rows tie
// exactly and the threadgroup shape never changes a bit. Any activation
// element works here (`lib/routed/routed.h` needs a 16-bit one).
//
// This file is independent of any entry ABI.

#include "../projection/projection.h"

namespace routing {

typedef element::Act Act;

// Threads of the stage launch whose RMS sum order every GEMV threadgroup
// reproduces.
constant constexpr uint stage_threads = 256;
constant constexpr uint stage_simdgroups = stage_threads / 32;

// The RMS inverses of `rows` rows of `residual` (`columns` columns) in the
// stage launch's sum order: 256 strided partials, simdgroup sums, then the
// eight simdgroup partials in order. Item (row, part) is part `part`'s
// simdgroup sum, at parts[row * stage_simdgroups + part]; the items spread
// over every simdgroup. Ends with the inverses published to the threadgroup.
inline void inverses(device const float *residual, ulong residual0, ulong residual1, uint columns, float eps,
    uint rows, threadgroup float *parts, threadgroup float *inverse, uint simdgroups, uint simd, uint lane,
    uint tid) {
    for (uint item = simd; item < rows * stage_simdgroups; item += simdgroups) {
        const uint row = item / stage_simdgroups, part = item % stage_simdgroups;
        float squares = 0.0f;
        for (uint source = part * 32 + lane; source < columns; source += stage_threads) {
            const float value = residual[ulong(row) * residual0 + source * residual1];
            squares = metal::fma(value, value, squares);
        }
        squares = simd_sum(squares);
        if (lane == 0)
            parts[row * stage_simdgroups + part] = squares;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < rows) {
        float total = 0.0f;
        for (uint part = 0; part < stage_simdgroups; ++part)
            total += parts[tid * stage_simdgroups + part];
        inverse[tid] = metal::rsqrt(total / float(columns) + eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

// The router input of row m at `source`: round_A(r * inverse * norm), with
// the norm's element N.
template <typename N>
struct Rows {
    static_assert(Act::bytes == 2, "the Metal router requires a bf16 or f16 activation element");
    typedef Act activation;
    device const float *residual;
    ulong residual0, residual1;
    device const uchar *norm;
    ulong norm0;
    threadgroup const float *inverses;
    uint columns;
    float value(uint m, ulong source) const {
        return Act::round(residual[ulong(m) * residual0 + source * residual1] * inverses[m]
            * element::at<N>(norm, source * norm0));
    }
    // The GEMV prologue: columns at or past `columns` are zero.
    void load8(uint m, uint k, float, thread float4 &even, thread float4 &odd) const {
        float v[8];
        PROJECTION_UNROLL
        for (uint i = 0; i < 8; ++i)
            v[i] = k + i < columns ? value(m, ulong(k + i)) : 0.0f;
        even = float4(v[0], v[2], v[4], v[6]);
        odd = float4(v[1], v[3], v[5], v[7]);
    }
};

// Logit column first + n of row m, rows `pitch` apart.
struct Logits {
    device float *logits;
    ulong pitch;
    ulong first;
    void store(uint m, uint n, float value) const { logits[ulong(m) * pitch + first + n] = value; }
};

// A dense row packet loaded whole when the GEMV loads it (H % 32 == 0, so
// every packet is whole): the GEMV's register double buffer then prefetches
// the weights, where `packets::Dense` defers its loads to the decode.
template <typename E>
struct EagerDense {
    static constant constexpr uint groups = 1;
    static constant constexpr uint words = 2 * E::bytes;
    struct packet {
        uint4 w[words];
    };
    static packet load(device const uchar *row, packets::Rows16 layout, uint p) {
        device const uint4 *at = reinterpret_cast<device const uint4 *>(row + layout.codes + ulong(p) * 32ul * E::bytes);
        packet k;
        PROJECTION_UNROLL
        for (uint i = 0; i < words; ++i)
            k.w[i] = at[i];
        return k;
    }
    static void codes(thread const packet &k, uint step, thread float4 &even, thread float4 &odd) {
        E::split8(k.w[step], even, odd);
    }
    static float scale(thread const packet &, uint) { return 1.0f; }
    static float bias(thread const packet &, uint) { return 0.0f; }
};
template <>
inline void EagerDense<element::F32>::codes(thread const packet &k, uint step, thread float4 &even,
    thread float4 &odd) {
    const float4 a = as_type<float4>(k.w[2 * step]), b = as_type<float4>(k.w[2 * step + 1]);
    even = float4(a.x, a.z, b.x, b.z);
    odd = float4(a.y, a.w, b.y, b.w);
}

// A router's decoder in the GEMV: dense rows load eagerly.
template <typename W>
struct Packets {
    typedef W type;
};
template <typename E>
struct Packets<packets::Dense<E>> {
    typedef EagerDense<E> type;
};

// Logit column `index` of weights `w` in one simdgroup that forms the router
// input as it reads. Loops over register arrays are unrolled, so no array
// lives in stack memory.
template <typename W, typename In>
inline void column(thread const In &in, thread const projection::Weights<W> &w, uint index,
    thread const Logits &out, uint rows, uint lane) {
    const uint k = in.columns;
    float sums[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    for (uint p = lane; p < k / 32; p += 32) {
        float4 we[4], wo[4];
        const typename W::packet packet = w.packet(index, p);
        PROJECTION_UNROLL
        for (uint step = 0; step < 4; ++step)
            projection::gemv_weights<W>(packet, step, we[step], wo[step]);
        PROJECTION_UNROLL
        for (uint m = 0; m < 8; ++m) {
            if (m < rows) {
                PROJECTION_UNROLL
                for (uint step = 0; step < 4; ++step) {
                    float4 xe, xo;
                    in.load8(m, 32 * p + 8 * step, 0.0f, xe, xo);
                    sums[m] = projection::gemv_step_dot(sums[m], we[step], wo[step], xe, xo);
                }
            }
        }
    }
    PROJECTION_UNROLL
    for (uint m = 0; m < 8; ++m) {
        if (m < rows) {
            const float sum = simd_sum(sums[m]);
            if (lane == 0)
                out.store(m, index, sum);
        }
    }
}

// Logit columns [SIMDGROUPS * group, ...) of the `count` rows of `w`, by
// activation width: 16-bit activations run the K1 GEMV body
// (`projection::gemv_body`, one 32-lane group per column) over the router
// input staged once per threadgroup (`shared`: the GEMV's staging memory);
// F32 activations, which the GEMV does not stage, one simdgroup per column
// (`column`). Both sum in the same order.
template <uint BYTES>
struct Gemv {
    template <typename W, typename In>
    static void columns(thread const In &in, thread const projection::Weights<W> &w, thread const Logits &out,
        uint rows, uint count, uint group, threadgroup uchar *shared, uint simdgroups, uint simd, uint lane) {
        PROJECTION_FOR_ROWS(rows,
            (projection::gemv_runtime<W, 1, MAXM, 32>(in, out, w, rows, count, in.columns, group, shared,
                simdgroups, simd, lane)));
    }
};
template <>
struct Gemv<4> {
    template <typename W, typename In>
    static void columns(thread const In &in, thread const projection::Weights<W> &w, thread const Logits &out,
        uint rows, uint count, uint group, threadgroup uchar *, uint simdgroups, uint simd, uint lane) {
        const uint index = group * simdgroups + simd;
        if (index < count)
            column(in, w, index, out, rows, lane);
    }
};

} // namespace routing
