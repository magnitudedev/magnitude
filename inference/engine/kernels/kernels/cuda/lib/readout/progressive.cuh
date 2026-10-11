// Progressive head readout (`readout.seismic`): the bit planes of a Q8_0 head
// in significance order, the levels of a certified selection over them, and
// the full exact pass. Selection rows: at most PROGRESSIVE_ROWS. Plane
// placement as in the Metal `lib/readout/progressive.h`:
//
//   top[v, w]       bits 7..4 of codes 8w..8w+7, code 8w + e in nibble e
//   bit3[v, g]      bit 3 of the codes of group g, code 32g + 8q + e in bit 4e + q
//   rest[v, g, j]   bit 2 - j, placed as in bit3
//   scales[v, g]    the f16 scale of group g
//
// The exact level and the full pass project a vocabulary row with one routine
// (`exact`, explicit roundings), lane l summing groups l, l + 32, ... in
// order and the warp reducing in one fixed tree, so a certified selection's
// exact logits equal the full pass's bit for bit. View comparisons keep a
// relative margin of PROGRESSIVE_MARGIN, so F32 rounding only keeps extra rows.
#include "../projection/projection.cuh"
#include "../core/gumbel.cuh"

#define PROGRESSIVE_ROWS 8u
#define PROGRESSIVE_CHUNK 1024u
#define PROGRESSIVE_MARGIN 3.814697265625e-06f
// The largest Gumbel noise the sampler draws: -log(-log u) at the largest
// uniform u = 1 - 2^-24, about 16.64.
#define PROGRESSIVE_NOISE 16.65f

namespace progressive {

using projection::Act;
using projection::u32;
using projection::u64;
using projection::u8;

// Order-preserving key of an F32 (zero is below every key), and back.
__device__ __forceinline__ u32 key(float value) {
    const u32 bits = __float_as_uint(value);
    return (bits & 0x80000000u) ? ~bits : (bits | 0x80000000u);
}
__device__ __forceinline__ float value_of(u32 k) {
    return __uint_as_float((k & 0x80000000u) ? (k & 0x7fffffffu) : ~k);
}

__device__ __forceinline__ float infinity() { return __int_as_float(0x7f800000); }

// Quarter q of a bit plane word: the bit of code 8q + e in nibble e.
__device__ __forceinline__ u32 quarter(u32 plane, u32 q) {
    return (plane >> q) & 0x11111111u;
}

// Byte J of `word` as an F32, less `bias` - 2^23 (exact: the float 2^23 + byte, less bias).
template <int J> __device__ __forceinline__ float byte(u32 word, float bias) {
    return __int_as_float(__byte_perm(word, 0x4B000000u, 0x7440u | J)) - bias;
}

constexpr float UNSIGNED = 8388608.0f;  // 2^23
constexpr float SIGNED = 8388736.0f;    // 2^23 + 128

// The 32 activations of one group, as pairs (2p, 2p + 1).
struct X {
    float2 pair[16];
};

// sum_e (code_e) x_e over quarter q of a group from even/odd byte words.
template <bool SIGN>
__device__ __forceinline__ float quarter_dot(u32 even, u32 odd, const X &x, u32 q, float dot) {
    constexpr float bias = SIGN ? SIGNED : UNSIGNED;
    dot = seismic_fma_rn(byte<0>(even, bias), x.pair[4 * q + 0].x, dot);
    dot = seismic_fma_rn(byte<0>(odd, bias), x.pair[4 * q + 0].y, dot);
    dot = seismic_fma_rn(byte<1>(even, bias), x.pair[4 * q + 1].x, dot);
    dot = seismic_fma_rn(byte<1>(odd, bias), x.pair[4 * q + 1].y, dot);
    dot = seismic_fma_rn(byte<2>(even, bias), x.pair[4 * q + 2].x, dot);
    dot = seismic_fma_rn(byte<2>(odd, bias), x.pair[4 * q + 2].y, dot);
    dot = seismic_fma_rn(byte<3>(even, bias), x.pair[4 * q + 3].x, dot);
    dot = seismic_fma_rn(byte<3>(odd, bias), x.pair[4 * q + 3].y, dot);
    return dot;
}

// One group's planes.
struct Group {
    uint4 high;
    u32 bit3, rest0, rest1, rest2;
    float scale;
};

// A group's 32 decoded codes in activation order (codes 2p and 2p + 1 beside
// pair p). Decoded once per vocabulary row, they serve every output row.
struct Codes {
    float value[32];
};

// Quarter q's codes from its even and odd byte words.
template <bool SIGN> __device__ __forceinline__ void decode(u32 even, u32 odd, u32 q, Codes &codes) {
    constexpr float bias = SIGN ? SIGNED : UNSIGNED;
    codes.value[8 * q + 0] = byte<0>(even, bias);
    codes.value[8 * q + 1] = byte<0>(odd, bias);
    codes.value[8 * q + 2] = byte<1>(even, bias);
    codes.value[8 * q + 3] = byte<1>(odd, bias);
    codes.value[8 * q + 4] = byte<2>(even, bias);
    codes.value[8 * q + 5] = byte<2>(odd, bias);
    codes.value[8 * q + 6] = byte<3>(even, bias);
    codes.value[8 * q + 7] = byte<3>(odd, bias);
}

// The group's exact signed codes c = u - 128.
__device__ __forceinline__ Codes exact_codes(const Group &group) {
    const u32 high[4] = {group.high.x, group.high.y, group.high.z, group.high.w};
    Codes codes;
#pragma unroll
    for (u32 q = 0; q < 4; ++q) {
        const u32 low = (quarter(group.bit3, q) << 3) | (quarter(group.rest0, q) << 2) |
                        (quarter(group.rest1, q) << 1) | quarter(group.rest2, q);
        decode<true>(((high[q] & 0x0f0f0f0fu) << 4) | (low & 0x0f0f0f0fu),
                     (high[q] & 0xf0f0f0f0u) | ((low >> 4) & 0x0f0f0f0fu), q, codes);
    }
    return codes;
}

// The group's top nibbles H, 0..15.
__device__ __forceinline__ Codes top_codes(const uint4 &words) {
    const u32 w[4] = {words.x, words.y, words.z, words.w};
    Codes codes;
#pragma unroll
    for (u32 q = 0; q < 4; ++q)
        decode<false>(w[q] & 0x0f0f0f0fu, (w[q] >> 4) & 0x0f0f0f0fu, q, codes);
    return codes;
}

// sum_i code_i x_i over a group, in activation order.
__device__ __forceinline__ float codes_dot(const Codes &codes, const X &x) {
    float dot = 0.0f;
#pragma unroll
    for (u32 p = 0; p < 16; ++p) {
        dot = seismic_fma_rn(codes.value[2 * p], x.pair[p].x, dot);
        dot = seismic_fma_rn(codes.value[2 * p + 1], x.pair[p].y, dot);
    }
    return dot;
}

// s * sum_i c_i x_i over a group from its exact codes: the exact level's and
// the full pass's one projection.
__device__ __forceinline__ float exact(float scale, const Codes &codes, const X &x) {
    return __fmul_rn(scale, codes_dot(codes, x));
}

// The warp's sum in one fixed tree.
__device__ __forceinline__ float warp_sum(float value) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        value = __fadd_rn(value, __shfl_xor_sync(0xffffffffu, value, offset));
    return value;
}

// Group g of row m of `features` (A, row stride `stride` elements).
__device__ __forceinline__ X features_group(const u8 *features, u64 stride, u32 m, u32 g) {
    const uint4 *words = reinterpret_cast<const uint4 *>(features + (m * stride + 32ull * g) * Act::bytes);
    X x;
#pragma unroll
    for (u32 q = 0; q < 4; ++q) {
        const uint4 w = words[q];
        x.pair[4 * q + 0] = Act::unpack2(w.x);
        x.pair[4 * q + 1] = Act::unpack2(w.y);
        x.pair[4 * q + 2] = Act::unpack2(w.z);
        x.pair[4 * q + 3] = Act::unpack2(w.w);
    }
    return x;
}

// The selection side of the output rows.
struct Selection {
    const u32 *draws;
    u64 draw_row, draw_word;
    const float *temperature;
    u64 temperature_stride;
    const u32 *mask;
    u64 mask_row, mask_word;
    const int *constrained;
    u64 constrained_stride;

    __device__ __forceinline__ bool competes(u32 m, u32 v) const {
        return constrained[m * constrained_stride] == 0 ||
               ((mask[m * mask_row + (u64)(v / 32u) * mask_word] >> (v % 32u)) & 1u) != 0u;
    }
    __device__ __forceinline__ float divisor(u32 m) const { return temperature[m * temperature_stride]; }
    __device__ __forceinline__ float score(u32 m, u32 v, float logit) const {
        return __fadd_rn(__fdiv_rn(logit, divisor(m)), gumbel::noise(v, draws + m * draw_row, draw_word));
    }
    __device__ __forceinline__ bool survives(u32 m, u32 v, float logit, float reach, float threshold) const {
        if (!(logit > -infinity()) || !competes(m, v))
            return false;
        return score(m, v, __fadd_rn(logit, reach)) >= threshold - PROGRESSIVE_MARGIN * (1.0f + fabsf(threshold));
    }
};

// Each row's largest lower-bound score into `bounds` (sync scratch:
// PROGRESSIVE_ROWS keys, then the arrival counter); the last block publishes
// max(floor, raised), or -inf where the row's length is not finite, and
// clears the keys.
__device__ __forceinline__ void publish(const u32 *lowest, u32 *bounds, u32 rows, const float *floor, u64 floor_stride,
                                        const float *lengths, u64 length_stride, float *threshold,
                                        u64 threshold_stride) {
    __shared__ u32 last;
    __syncthreads();
    if (threadIdx.x < rows)
        atomicMax(&bounds[threadIdx.x], lowest[threadIdx.x]);
    __syncthreads();
    if (threadIdx.x == 0) {
        __threadfence();
        const bool is_last = atomicAdd(&bounds[PROGRESSIVE_ROWS], 1u) == gridDim.x - 1;
        if (is_last) {
            __threadfence();
            atomicExch(&bounds[PROGRESSIVE_ROWS], 0u);
        }
        last = is_last;
    }
    __syncthreads();
    if (last && threadIdx.x < rows) {
        const u32 k = atomicExch(&bounds[threadIdx.x], 0u);
        const float raised = k == 0u ? -infinity() : value_of(k);
        const float floored = floor ? fmaxf(floor[threadIdx.x * floor_stride], raised) : raised;
        threshold[threadIdx.x * threshold_stride] =
            isfinite(lengths[threadIdx.x * length_stride]) ? floored : -infinity();
    }
}

// Row blockIdx.x of the prologue into `features` (A); with GROUPS also its
// length (over the rounded values) and per-group sums ([rows][d / 32]).
template <bool GROUPS, class Pro>
__device__ __forceinline__ void form(const Pro &pro, u32 d, u8 *features, u64 feature_stride, float *lengths,
                                     u64 length_stride, float *groups) {
    __shared__ float factors[Pro::FACTORS > 0 ? Pro::FACTORS : 1];
    __shared__ float partials[32];
    const u32 m = blockIdx.x;
    pro.prepare_row(factors, m, partials);
    __syncthreads();
    const u32 lane = threadIdx.x % 32, warps = blockDim.x / 32;
    float squares = 0.0f;
    for (u32 g = threadIdx.x / 32; g < d / 32; g += warps) {
        const float value = pro.value(factors, m, 32ull * g + lane);
        element::put<Act>(features, m * feature_stride + 32ull * g + lane, value);
        if constexpr (GROUPS) {
            squares = seismic_fma_rn(value, value, squares);
            const float sum = warp_sum(value);
            if (lane == 0)
                groups[m * (d / 32) + g] = sum;
        }
    }
    if constexpr (GROUPS) {
        __syncthreads();
        const float total = reduce::group_sum(squares, partials);
        if (threadIdx.x == 0)
            lengths[m * length_stride] = sqrtf(total);
    }
}

struct Planes {
    const u32 *top;
    u64 top_stride;
    const u32 *bit3;
    u64 bit3_stride;
    const u32 *rest;
    u64 rest_row, rest_group, rest_plane;
    const unsigned short *scales;
    u64 scale_stride;

    __device__ __forceinline__ uint4 high(u32 v, u32 g) const {
        return *reinterpret_cast<const uint4 *>(top + v * top_stride + 4u * g);
    }
    __device__ __forceinline__ float scale(u32 v, u32 g) const {
        return seismic_f16_to_f32(scales[v * scale_stride + g]);
    }
    __device__ __forceinline__ u32 third(u32 v, u32 g) const { return bit3[v * bit3_stride + g]; }
    __device__ __forceinline__ Group group(u32 v, u32 g) const {
        const u32 *low = rest + v * rest_row + g * rest_group;
        return Group{high(v, g), third(v, g), low[0], low[rest_plane], low[2 * rest_plane], scale(v, g)};
    }
};

// The vocabulary GEMV over every row: ROWS vocabulary rows per warp, lane l
// reading group l of each 1,024-column chunk (the chunk's activations staged
// in `staged`, PROGRESSIVE_ROWS x PROGRESSIVE_CHUNK A values of shared memory,
// quarter-major so a warp's 16-byte reads are conflict-free; `sums` holds
// PROGRESSIVE_ROWS x 32 group sums, `lowest` PROGRESSIVE_ROWS keys). EXACT: the exact logits. Otherwise the 4-bit view
// `s * (16 H - 120.5)`, whose lower-bound scores raise each row's threshold.
template <u32 MAXM, u32 ROWS, bool EXACT>
__device__ __forceinline__ void scan(const u8 *features, u64 feature_stride, const float *groups, const Planes &planes,
                                     const float *radius, u64 radius_stride, const float *lengths, u64 length_stride,
                                     const Selection &selection, float *logits, u64 logit_row, u64 logit_col,
                                     float *threshold, u64 threshold_stride, u32 *bounds, u32 rows, u32 vocabulary,
                                     u32 d, uint4 *staged, float *sums, u32 *lowest) {
    const u32 lane = threadIdx.x % 32, warps = blockDim.x / 32;
    if (!EXACT && threadIdx.x < PROGRESSIVE_ROWS)
        lowest[threadIdx.x] = 0u;
    const u32 first = (blockIdx.x * warps + threadIdx.x / 32) * ROWS;
    float acc[ROWS][MAXM];
#pragma unroll
    for (u32 r = 0; r < ROWS; ++r)
#pragma unroll
        for (u32 m = 0; m < MAXM; ++m)
            acc[r][m] = 0.0f;
    for (u32 chunk = 0; chunk < d; chunk += PROGRESSIVE_CHUNK) {
        const u32 width = min(PROGRESSIVE_CHUNK, d - chunk);
        __syncthreads();
        for (u32 item = threadIdx.x; item < rows * (width / 8); item += blockDim.x) {
            const u32 m = item / (width / 8), w = item % (width / 8);
            staged[(m * 4 + w % 4) * 32 + w / 4] =
                *reinterpret_cast<const uint4 *>(features + (m * feature_stride + chunk + 8ull * w) * Act::bytes);
        }
        if constexpr (!EXACT)
            for (u32 item = threadIdx.x; item < rows * (width / 32); item += blockDim.x) {
                const u32 m = item / (width / 32), g = item % (width / 32);
                sums[m * 32 + g] = groups[m * (d / 32) + chunk / 32 + g];
            }
        __syncthreads();
        if (32 * lane >= width)
            continue;
        const u32 g = chunk / 32 + lane;
        // Every row's words first, so their loads are in flight together.
        Group group[ROWS];
#pragma unroll
        for (u32 r = 0; r < ROWS; ++r) {
            const u32 v = min(first + r, vocabulary - 1);
            if constexpr (EXACT) {
                group[r] = planes.group(v, g);
            } else {
                group[r].high = planes.high(v, g);
                group[r].scale = planes.scale(v, g);
            }
        }
#pragma unroll
        for (u32 r = 0; r < ROWS; ++r) {
            const Codes codes = EXACT ? exact_codes(group[r]) : top_codes(group[r].high);
#pragma unroll
            for (u32 m = 0; m < MAXM; ++m) {
                if (m >= rows)
                    break;
                X x;
#pragma unroll
                for (u32 q = 0; q < 4; ++q) {
                    const uint4 w = staged[(m * 4 + q) * 32 + lane];
                    x.pair[4 * q + 0] = Act::unpack2(w.x);
                    x.pair[4 * q + 1] = Act::unpack2(w.y);
                    x.pair[4 * q + 2] = Act::unpack2(w.z);
                    x.pair[4 * q + 3] = Act::unpack2(w.w);
                }
                if constexpr (EXACT) {
                    acc[r][m] = __fadd_rn(acc[r][m], exact(group[r].scale, codes, x));
                } else {
                    const float view = seismic_fma_rn(16.0f, codes_dot(codes, x),
                                                      __fmul_rn(-120.5f, sums[m * 32 + lane]));
                    acc[r][m] = seismic_fma_rn(group[r].scale, view, acc[r][m]);
                }
            }
        }
    }
#pragma unroll
    for (u32 r = 0; r < ROWS; ++r) {
        const u32 v = first + r;
#pragma unroll
        for (u32 m = 0; m < MAXM; ++m) {
            if (m >= rows)
                break;
            const float logit = warp_sum(acc[r][m]);
            if (v >= vocabulary || lane != m)
                continue;
            logits[m * logit_row + v * logit_col] = logit;
            if (!EXACT && selection.competes(m, v)) {
                // A score is its noise-free score plus noise of at most
                // PROGRESSIVE_NOISE: a row that cannot reach the threshold so
                // far cannot raise it, so it skips the noise.
                const float lower =
                    __fsub_rn(logit, __fmul_rn(radius[v * radius_stride], lengths[m * length_stride]));
                const u32 best = *reinterpret_cast<volatile const u32 *>(&lowest[m]);
                if (best == 0u || __fdiv_rn(lower, selection.divisor(m)) + PROGRESSIVE_NOISE >= value_of(best))
                    atomicMax(&lowest[m], key(selection.score(m, v, lower)));
            }
        }
    }
    if constexpr (!EXACT)
        publish(lowest, bounds, rows, nullptr, 0, lengths, length_stride, threshold, threshold_stride);
}

// A later level: one warp per 32 vocabulary rows. Lane l decides row l for
// every output row (the previous level's upper-bound score against `floor`,
// its threshold; −inf elsewhere); the warp then projects each survivor, lane l
// summing groups l, l + 32, ... REFINE: the 5-bit view, the 4-bit logit plus
// `s * (8 b3 - 4) x`, whose lower-bound scores raise the next threshold.
// Otherwise: the exact logit.
template <bool REFINE>
__device__ __forceinline__ void gather(const u8 *features, u64 feature_stride, const Planes &planes,
                                       const float *radius_previous, const float *radius_next, u64 radius_stride,
                                       const float *coarse, u64 coarse_row, u64 coarse_col, const float *floor,
                                       u64 floor_stride, const float *lengths, u64 length_stride,
                                       const Selection &selection, float *out, u64 out_row, u64 out_col,
                                       float *threshold, u64 threshold_stride, u32 *bounds, u32 rows, u32 vocabulary,
                                       u32 d) {
    __shared__ u32 lowest[PROGRESSIVE_ROWS];
    const u32 lane = threadIdx.x % 32, warps = blockDim.x / 32;
    if (REFINE && threadIdx.x < PROGRESSIVE_ROWS)
        lowest[threadIdx.x] = 0u;
    if (REFINE)
        __syncthreads();
    const u32 first = (blockIdx.x * warps + threadIdx.x / 32) * 32;
    const u32 n = first + lane;
    u32 candidate = 0u;
    if (n < vocabulary)
        for (u32 m = 0; m < rows; ++m) {
            const float reach = __fmul_rn(radius_previous[n * radius_stride], lengths[m * length_stride]);
            if (selection.survives(m, n, coarse[m * coarse_row + n * coarse_col], reach, floor[m * floor_stride]))
                candidate |= 1u << m;
            else
                out[m * out_row + n * out_col] = -infinity();
        }
    const u32 groups = d / 32;
    for (u32 pending = __ballot_sync(0xffffffffu, candidate != 0u); pending != 0u; pending &= pending - 1u) {
        const u32 i = __ffs(pending) - 1;
        const u32 v = first + i;
        const u32 which = __shfl_sync(0xffffffffu, candidate, i);
        for (u32 m = 0; m < rows; ++m) {
            if ((which >> m & 1u) == 0u)
                continue;
            float acc = 0.0f;
            for (u32 g = lane; g < groups; g += 32) {
                const X x = features_group(features, feature_stride, m, g);
                if constexpr (REFINE) {
                    const u32 plane = planes.third(v, g);
                    float dot = 0.0f, sum = 0.0f;
#pragma unroll
                    for (u32 q = 0; q < 4; ++q) {
                        const u32 word = quarter(plane, q);
                        dot = quarter_dot<false>(word & 0x0f0f0f0fu, (word >> 4) & 0x0f0f0f0fu, x, q, dot);
#pragma unroll
                        for (u32 p = 0; p < 4; ++p)
                            sum = __fadd_rn(sum, __fadd_rn(x.pair[4 * q + p].x, x.pair[4 * q + p].y));
                    }
                    acc = seismic_fma_rn(planes.scale(v, g), seismic_fma_rn(8.0f, dot, __fmul_rn(-4.0f, sum)), acc);
                } else {
                    const Group group = planes.group(v, g);
                    acc = __fadd_rn(acc, exact(group.scale, exact_codes(group), x));
                }
            }
            acc = warp_sum(acc);
            if (lane != 0)
                continue;
            const float logit = REFINE ? __fadd_rn(coarse[m * coarse_row + v * coarse_col], acc) : acc;
            out[m * out_row + v * out_col] = logit;
            if (REFINE && selection.competes(m, v))
                atomicMax(&lowest[m],
                          key(selection.score(m, v, __fsub_rn(logit, __fmul_rn(radius_next[v * radius_stride],
                                                                                lengths[m * length_stride])))));
        }
    }
    if constexpr (REFINE)
        publish(lowest, bounds, rows, floor, floor_stride, lengths, length_stride, threshold, threshold_stride);
}

} // namespace progressive
