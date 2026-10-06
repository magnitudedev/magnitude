// Progressive head readout (`readout.seismic`): the bit planes of a Q8_0 head
// in significance order, the levels of a certified selection over them, and
// the full exact pass. Selection rows: at most PROGRESSIVE_ROWS.
//
//   top[v, w]       bits 7..4 of codes 8w..8w+7, code 8w + e in nibble e
//   bit3[v, g]      bit 3 of the codes of group g, code 32g + 8q + e in bit 4e + q
//   rest[v, g, j]   bit 2 - j, placed as in bit3
//   scales[v, g]    the f16 scale of group g (codes 32g..32g+31)
//
// The placement makes every quarter (8 codes, one uint4 of activations) a
// shift and mask away: `(plane >> q) & 0x11111111` holds the bit of code
// 8q + e in nibble e, as `top` holds its high bits.
//
// The top level and the full pass are the projection library's GEMV,
// batched GEMV and GEMM over the views' packets; the later levels gather the
// survivors' remaining planes. The views are bounded (`radius`) rather than
// exact; their comparisons keep a relative margin of PROGRESSIVE_MARGIN, so
// F32 rounding only ever keeps extra rows.
#include "../projection/projection.h"
#include "../core/arrive.h"
#include "../core/gumbel.h"

#define PROGRESSIVE_ROWS 8u
#define PROGRESSIVE_MARGIN 0x1p-18f
// The largest Gumbel noise the sampler draws: -log(-log u) at the largest
// uniform u = 1 - 2^-24, about 16.64.
#define PROGRESSIVE_NOISE 16.65f
// Threadgroup words (uint4) a later level stages its features in: four rows
// of 2,048 A values.
#define PROGRESSIVE_STAGE_WORDS 1024u

// The two views of a progressive head as projection-library packets, so the
// library's GEMV, batched GEMV and GEMM project its planes. A packet is one
// group of 32 codes, consumed in four steps (quarters) of 8.
namespace packets {

// The 4-bit view: the top plane's nibbles H, in q4k's low-nibble order, at
// scale 16 s and bias -120.5 s (the code's missing bits at their midpoint).
// Both are exact in F32 (an f16 scale times a small integer).
struct ProgressiveTop {
    static constant constexpr uint groups = 1;
    struct packet {
        uint4 low;
        float scale;
        float bias;
    };
    static void codes(thread const packet &k, uint step, thread float4 &even, thread float4 &odd) {
        uint e, o;
        split_nibbles(k.low[step], e, o);
        even = unsigned_bytes(e);
        odd = unsigned_bytes(o);
    }
    static float2 pair(thread const packet &k, uint step, uint j) {
        uint b = (k.low[step] >> (8u * j)) & 0xffu;
        return code_pair((b & 15u) | ((b >> 4) << 16), 1024.0h);
    }
    static float scale(thread const packet &k, uint) { return k.scale; }
    static float bias(thread const packet &k, uint) { return k.bias; }
    static float value(thread const packet &k, uint, float code) { return metal::fma(k.scale, code, k.bias); }
};

// The exact view: the offset-binary codes u of every plane, as c = u - 128,
// at scale s.
struct ProgressiveExact {
    static constant constexpr uint groups = 1;
    struct packet {
        uint4 high;
        uint4 low;  // bit 3, then bits 2, 1, 0
        float scale;
    };
    // Step q's code bytes u: codes 8q, 8q + 2, ... in `even`, the others in
    // `odd`.
    static void bytes(thread const packet &k, uint step, thread uint &even, thread uint &odd) {
        const uint low = (((k.low.x >> step) & 0x11111111u) << 3) | (((k.low.y >> step) & 0x11111111u) << 2)
            | (((k.low.z >> step) & 0x11111111u) << 1) | ((k.low.w >> step) & 0x11111111u);
        const uint high = k.high[step];
        even = ((high & 0x0f0f0f0fu) << 4) | (low & 0x0f0f0f0fu);
        odd = (high & 0xf0f0f0f0u) | ((low >> 4) & 0x0f0f0f0fu);
    }
    static void codes(thread const packet &k, uint step, thread float4 &even, thread float4 &odd) {
        uint e, o;
        bytes(k, step, e, o);
        even = byte_floats(e, 1152.0h);
        odd = byte_floats(o, 1152.0h);
    }
    static float2 pair(thread const packet &k, uint step, uint j) {
        uint e, o;
        bytes(k, step, e, o);
        return code_pair(((e >> (8u * j)) & 0xffu) | (((o >> (8u * j)) & 0xffu) << 16), 1152.0h);
    }
    static float scale(thread const packet &k, uint) { return k.scale; }
    static float bias(thread const packet &, uint) { return 0.0f; }
    static float value(thread const packet &k, uint, float code) { return k.scale * code; }
};

} // namespace packets

namespace projection {

// A progressive head's planes as the library's weights of each view.
template <>
struct Weights<packets::ProgressiveTop> {
    device const uint *top;
    ulong top_stride;
    device const half *scales;
    ulong scale_stride;
    packets::ProgressiveTop::packet packet(uint n, uint p) const {
        const float s = float(scales[ulong(n) * scale_stride + p]);
        return {*reinterpret_cast<device const uint4 *>(top + ulong(n) * top_stride + 4u * p), 16.0f * s,
            -120.5f * s};
    }
    // The planes keep no coefficient run (every packet has its own scale)
    // and no row tiles.
    packets::Block<packets::ProgressiveTop>::state run(uint, uint) const { return {}; }
    packets::ProgressiveTop::packet packet(uint n, uint p,
        thread packets::Block<packets::ProgressiveTop>::state &) const {
        return packet(n, p);
    }
    uint tile() const { return 1u; }
    // A row is located by its index.
    typedef uint located;
    located locate(uint n) const { return n; }
    template <uint R>
    bool together(uint) const { return false; }
    static located after(located at, uint r) { return at + r; }
};

template <>
struct Weights<packets::ProgressiveExact> {
    device const uint *top;
    ulong top_stride;
    device const uint *bit3;
    ulong bit3_stride;
    device const uint *rest;
    ulong rest_row, rest_group, rest_plane;
    device const half *scales;
    ulong scale_stride;
    packets::ProgressiveExact::packet packet(uint n, uint p) const {
        device const uint *low = rest + ulong(n) * rest_row + ulong(p) * rest_group;
        return {*reinterpret_cast<device const uint4 *>(top + ulong(n) * top_stride + 4u * p),
            uint4(bit3[ulong(n) * bit3_stride + p], low[0], low[rest_plane], low[2u * rest_plane]),
            float(scales[ulong(n) * scale_stride + p])};
    }
    // The planes keep no coefficient run: every packet has its own scale.
    packets::Block<packets::ProgressiveExact>::state run(uint, uint) const { return {}; }
    packets::ProgressiveExact::packet packet(uint n, uint p,
        thread packets::Block<packets::ProgressiveExact>::state &) const {
        return packet(n, p);
    }
    uint tile() const { return 1u; }
    typedef uint located;
    located locate(uint n) const { return n; }
    template <uint R>
    bool together(uint) const { return false; }
    static located after(located at, uint r) { return at + r; }
};

} // namespace projection

namespace progressive {

typedef element::Act activation;
static_assert(activation::bytes == 2, "the progressive readout requires a bf16 or f16 activation element");

// Order-preserving key of an F32 (larger float, larger key; every key of a
// non-NaN float is nonzero, so zero is below every key), and back.
inline uint key(float value) {
    const uint bits = as_type<uint>(value);
    return (bits & 0x80000000u) ? ~bits : (bits | 0x80000000u);
}
inline float value_of(uint k) {
    return as_type<float>((k & 0x80000000u) ? (k & 0x7fffffffu) : ~k);
}

// Quarter q of a bit plane word: the bit of code 8q + e in nibble e.
inline uint quarter(uint plane, uint q) {
    return (plane >> q) & 0x11111111u;
}

// The even (0, 2, 4, 6) and odd codes of a nibble word, as 0..15.
inline void nibbles(uint word, thread float4 &even, thread float4 &odd) {
    even = float4(as_type<uchar4>(word & 0x0f0f0f0fu));
    odd = float4(as_type<uchar4>((word >> 4) & 0x0f0f0f0fu));
}

// One group's planes.
struct Group {
    uint4 high;
    uint bit3, rest0, rest1, rest2;
    float scale;
};

// A group's 32 decoded codes, by quarter: the even (8q, 8q + 2, ...) and odd
// codes. Decoded once per vocabulary row, they serve every output row.
struct Codes {
    float4 even[4], odd[4];
};

// The group's exact signed codes c = u - 128.
inline Codes exact_codes(thread const Group &group) {
    Codes codes;
    for (uint q = 0; q < 4u; ++q) {
        const uint low = (quarter(group.bit3, q) << 3) | (quarter(group.rest0, q) << 2)
            | (quarter(group.rest1, q) << 1) | quarter(group.rest2, q);
        const uint high = group.high[q];
        codes.even[q] = float4(as_type<char4>((((high & 0x0f0f0f0fu) << 4) | (low & 0x0f0f0f0fu)) ^ 0x80808080u));
        codes.odd[q] = float4(as_type<char4>(((high & 0xf0f0f0f0u) | ((low >> 4) & 0x0f0f0f0fu)) ^ 0x80808080u));
    }
    return codes;
}

// sum_i code_i x_i over a group, quarter by quarter.
inline float codes_dot(thread const Codes &codes, thread const float4 (&xe)[4], thread const float4 (&xo)[4]) {
    float dot = 0.0f;
    for (uint q = 0; q < 4u; ++q)
        dot += metal::dot(codes.even[q], xe[q]) + metal::dot(codes.odd[q], xo[q]);
    return dot;
}

// s * sum_i c_i x_i over a group from its exact codes: the exact level's and
// the full pass's one projection.
inline float exact(float scale, thread const Codes &codes, thread const float4 (&xe)[4],
    thread const float4 (&xo)[4]) {
    return scale * codes_dot(codes, xe, xo);
}

// The selection side of the output rows: sampler draws, score divisors and
// competition masks.
struct Selection {
    device const uint *draws;
    ulong draw_row, draw_word;
    device const float *temperature;
    ulong temperature_stride;
    device const uint *mask;
    ulong mask_row, mask_word;
    device const int *constrained;
    ulong constrained_stride;

    bool competes(uint m, uint v) const {
        return constrained[m * constrained_stride] == 0
            || ((mask[m * mask_row + ulong(v / 32u) * mask_word] >> (v % 32u)) & 1u) != 0u;
    }
    float divisor(uint m) const {
        return temperature[ulong(m) * temperature_stride];
    }
    float score(uint m, uint v, float logit) const {
        return logit / divisor(m) + gumbel::noise(v, draws + ulong(m) * draw_row, draw_word);
    }
    // Whether a competing row whose view logit is within `reach` of its
    // exact logit can reach `threshold`.
    bool survives(uint m, uint v, float logit, float reach, float threshold) const {
        if (!(logit > -INFINITY) || !competes(m, v))
            return false;
        return score(m, v, logit + reach) >= threshold - PROGRESSIVE_MARGIN * (1.0f + metal::abs(threshold));
    }
};

// Each row's largest lower-bound score, raised by every threadgroup; the
// last to arrive publishes max(floor, raised), or -inf where the row's length
// is not finite (no bound holds), and clears the keys. `bounds`: sync scratch,
// PROGRESSIVE_ROWS keys then the arrival counter.
inline void publish(threadgroup atomic_uint *lowest, device atomic_uint *bounds, threadgroup uint *last, uint tiles,
    uint rows, uint tid, device const float *floor, ulong floor_stride, device const float *lengths,
    ulong length_stride, device float *threshold, ulong threshold_stride) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < rows)
        atomic_fetch_max_explicit(&bounds[tid], atomic_load_explicit(&lowest[tid], memory_order_relaxed),
            memory_order_relaxed);
    if (arrive::last(&bounds[PROGRESSIVE_ROWS], tiles, last, tid) && tid < rows) {
        const uint k = atomic_load_explicit(&bounds[tid], memory_order_relaxed);
        const float raised = k == 0u ? -INFINITY : value_of(k);
        const float floored = floor ? metal::max(floor[tid * floor_stride], raised) : raised;
        threshold[tid * threshold_stride] = metal::isfinite(lengths[tid * length_stride]) ? floored : -INFINITY;
        atomic_store_explicit(&bounds[tid], 0u, memory_order_relaxed);
    }
}

// The four quarters of group g of row m of `features`.
inline void features_group(device const uchar *features, ulong stride, uint m, uint g, thread float4 (&xe)[4],
    thread float4 (&xo)[4]) {
    device const uint4 *x = reinterpret_cast<device const uint4 *>(
        reinterpret_cast<device const typename activation::storage *>(features) + ulong(m) * stride + 32u * g);
    for (uint q = 0; q < 4u; ++q)
        activation::split8(x[q], xe[q], xo[q]);
}

struct Planes {
    device const uint *top;
    ulong top_stride;
    device const uint *bit3;
    ulong bit3_stride;
    device const uint *rest;
    ulong rest_row, rest_group, rest_plane;
    device const half *scales;
    ulong scale_stride;

    uint4 high(uint v, uint g) const {
        return *reinterpret_cast<device const uint4 *>(top + ulong(v) * top_stride + 4u * g);
    }
    float scale(uint v, uint g) const {
        return float(scales[ulong(v) * scale_stride + g]);
    }
    uint third(uint v, uint g) const {
        return bit3[ulong(v) * bit3_stride + g];
    }
    Group group(uint v, uint g) const {
        device const uint *low = rest + ulong(v) * rest_row + ulong(g) * rest_group;
        return Group{high(v, g), third(v, g), low[0], low[rest_plane], low[2u * rest_plane], scale(v, g)};
    }
};

// A later level: one simdgroup per 32 vocabulary rows. Lane l decides row l
// for every output row: it survives when the previous level's upper-bound
// score reaches `floor` (that level's threshold); −inf elsewhere. The
// simdgroup then projects each survivor, lane l summing groups l, l + 32, ...
// REFINE: the 5-bit view, the 4-bit logit plus `s * (8 b3 - 4) x`, whose
// lower-bound scores raise the next threshold. Otherwise: the exact logit.
template <bool REFINE>
inline void gather(device const uchar *features, ulong feature_stride, thread const Planes &planes,
    device const float *radius_previous, device const float *radius_next, ulong radius_stride, device const float *coarse,
    ulong coarse_row, ulong coarse_col, device const float *floor, ulong floor_stride, device const float *lengths, ulong length_stride,
    thread const Selection &selection, device float *out, ulong out_row, ulong out_col, device float *threshold,
    ulong threshold_stride, device atomic_uint *bounds, uint rows, uint vocabulary, uint d,
    threadgroup atomic_uint *lowest, threadgroup uint *last, threadgroup uint4 *stage, uint tile, uint tiles, uint tid,
    uint threads, uint simdgroups, uint sg, uint lane) {
    if (REFINE && tid < PROGRESSIVE_ROWS)
        atomic_store_explicit(&lowest[tid], 0u, memory_order_relaxed);
    // The features in threadgroup memory when they fit, quarter-major
    // ([row][quarter][group]) so a simdgroup's reads are contiguous; every
    // survivor of the threadgroup reads them.
    const uint groups = d / 32u;
    const bool staged = rows * groups * 4u <= PROGRESSIVE_STAGE_WORDS;
    if (staged)
        for (uint item = tid; item < rows * groups * 4u; item += threads) {
            const uint m = item / (groups * 4u), w = item % (groups * 4u);
            stage[(m * 4u + w % 4u) * groups + w / 4u] = *reinterpret_cast<device const uint4 *>(
                reinterpret_cast<device const typename activation::storage *>(features) + ulong(m) * feature_stride
                + 8u * w);
        }
    if (REFINE || staged)
        threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint first = (tile * simdgroups + sg) * 32u;
    const uint n = first + lane;
    // Bit m: output row m keeps vocabulary row n.
    uint candidate = 0u;
    if (n < vocabulary)
        for (uint m = 0; m < rows; ++m) {
            const float reach = radius_previous[n * radius_stride] * lengths[m * length_stride];
            if (selection.survives(m, n, coarse[m * coarse_row + ulong(n) * coarse_col], reach,
                    floor[m * floor_stride]))
                candidate |= 1u << m;
            else
                out[m * out_row + ulong(n) * out_col] = -INFINITY;
        }
    for (uint pending = uint(uint64_t(simd_ballot(candidate != 0u))); pending != 0u; pending &= pending - 1u) {
        const uint i = ctz(pending);
        const uint v = first + i;
        const uint which = simd_shuffle(candidate, ushort(i));
        for (uint m = 0; m < rows; ++m) {
            if ((which >> m & 1u) == 0u)
                continue;
            float acc = 0.0f;
            for (uint g = lane; g < groups; g += 32u) {
                float4 xe[4], xo[4];
                if (staged)
                    for (uint q = 0; q < 4u; ++q)
                        activation::split8(stage[(m * 4u + q) * groups + g], xe[q], xo[q]);
                else
                    features_group(features, feature_stride, m, g, xe, xo);
                if (REFINE) {
                    const uint plane = planes.third(v, g);
                    float dot = 0.0f, sum = 0.0f;
                    for (uint q = 0; q < 4u; ++q) {
                        float4 be, bo;
                        nibbles(quarter(plane, q), be, bo);
                        dot += metal::dot(be, xe[q]) + metal::dot(bo, xo[q]);
                        sum += (xe[q].x + xe[q].y + xe[q].z + xe[q].w) + (xo[q].x + xo[q].y + xo[q].z + xo[q].w);
                    }
                    acc = metal::fma(planes.scale(v, g), metal::fma(8.0f, dot, -4.0f * sum), acc);
                } else {
                    acc += exact(planes.scale(v, g), exact_codes(planes.group(v, g)), xe, xo);
                }
            }
            acc = simd_sum(acc);
            if (lane != 0u)
                continue;
            const float logit = REFINE ? coarse[m * coarse_row + ulong(v) * coarse_col] + acc : acc;
            out[m * out_row + ulong(v) * out_col] = logit;
            if (REFINE && selection.competes(m, v))
                atomic_fetch_max_explicit(&lowest[m],
                    key(selection.score(m, v, logit - radius_next[v * radius_stride] * lengths[m * length_stride])),
                    memory_order_relaxed);
        }
    }
    if (REFINE)
        publish(lowest, bounds, last, tiles, rows, tid, floor, floor_stride, lengths, length_stride, threshold,
            threshold_stride);
}

} // namespace progressive
