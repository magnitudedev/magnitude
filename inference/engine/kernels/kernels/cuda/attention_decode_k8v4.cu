// attention_decode_k8v4 (M <= 8): `attention_decode`'s split-KV decode over
// affine K8/V4 history.
//
// L1 `attention_decode_k8v4_partial`, one block per (kv head, partition, row),
// partitioned and sliced over the query group exactly as the dense entry. History keys score as the sum over
// groups of scale * (q . code) + zero * sum(q) and values accumulate
// (p * scale) * code plus the bias sum(p * zero) of each lane's group, carried
// through the online softmax by the same factor as the accumulator (`absorb`),
// so no history element is decoded. The fresh span stays dense; the bias is
// folded into the accumulator before it. Partition 0 of a layer with fresh
// rows appends the row's key and value encoded. L2 `attention_decode_k8v4_merge`:
// the dense entry's merge and gate.

#include "lib/attention/attention.cuh"

namespace {

using attention::Act;
using attention::DPL;
using attention::G;
using attention::KV;
using attention::W;
using attention::u32;
using attention::u64;

typedef attention::LaneCodes<attention::KEY_BITS> KeyCodes;
typedef attention::LaneCodes<attention::VALUE_BITS> ValueCodes;

constexpr int WARPS = SEISMIC_TUNE_WARPS;
constexpr int PARTS = SEISMIC_TUNE_PARTS;
// The G query heads split into SLICES slices of H heads, as in
// `attention_decode`: warp w holds slice w % SLICES over key group w / SLICES.
constexpr int SLICES = SEISMIC_TUNE_SLICES;
constexpr int H = G / SLICES;
constexpr int GROUPS = WARPS / SLICES;
// History keys a warp keeps in flight: affine rows are 2.6x smaller than
// dense ones, so twice the dense batch keeps as many bytes in flight (fewer
// for heads wider than 256, whose lanes hold 16 dimensions per query head).
constexpr int TOKENS = DPL >= 16 ? 2 : (H >= 8 ? 4 : 8);
typedef attention::Heads<H> State;

// Absorb `count` (1..N) affine-coded keys and values, with this lane's group
// pairs `kc` and `vc`, into `state` and this lane's per-head `bias` (exp2
// domain; `q` scaled into it, `qsum` the sums of this lane's dimensions).
template <int N>
__device__ __forceinline__ void absorb(State &state, float (&bias)[H],
                                       const float (&q)[H][DPL], const float (&qsum)[H],
                                       const u32 (&k)[N][KeyCodes::words],
                                       const float2 (&kc)[N],
                                       const u32 (&v)[N][ValueCodes::words],
                                       const float2 (&vc)[N], int count) {
    float score[N][H];
#pragma unroll
    for (int t = 0; t < N; ++t) {
        float key[DPL];
#pragma unroll
        for (int d = 0; d < DPL; ++d) key[d] = KeyCodes::code(k[t], d);
#pragma unroll
        for (int h = 0; h < H; ++h) {
            float dot = 0.0f;
#pragma unroll
            for (int d = 0; d < DPL; ++d) dot = __fmaf_rn(q[h][d], key[d], dot);
            score[t][h] = seismic_warp_sum_f32(__fmaf_rn(kc[t].x, dot, kc[t].y * qsum[h]));
        }
    }
    float value[N][DPL];
#pragma unroll
    for (int t = 0; t < N; ++t)
#pragma unroll
        for (int d = 0; d < DPL; ++d) value[t][d] = ValueCodes::code(v[t], d);
#pragma unroll
    for (int h = 0; h < H; ++h) {
        float maximum = state.maximum[h];
#pragma unroll
        for (int t = 0; t < N; ++t)
            if (t < count) maximum = fmaxf(maximum, score[t][h]);
        // exp2(-inf) = 0 for the first keys of an empty state.
        const float carry = seismic_ex2_approx(state.maximum[h] - maximum);
        float denominator = state.denominator[h] * carry;
        float offset = bias[h] * carry;
#pragma unroll
        for (int d = 0; d < DPL; ++d) state.accumulator[h][d] *= carry;
#pragma unroll
        for (int t = 0; t < N; ++t) {
            if (t < count) {
                const float p = seismic_ex2_approx(score[t][h] - maximum);
                denominator += p;
                offset = __fmaf_rn(p, vc[t].y, offset);
                const float weight = p * vc[t].x;
#pragma unroll
                for (int d = 0; d < DPL; ++d)
                    state.accumulator[h][d] =
                        __fmaf_rn(weight, value[t][d], state.accumulator[h][d]);
            }
        }
        state.denominator[h] = denominator;
        state.maximum[h] = maximum;
        bias[h] = offset;
    }
}

// Fold the per-head bias into the accumulator (it then carries the whole
// attended sum), for the dense absorb or the publication.
__device__ __forceinline__ void fold(State &state, float (&bias)[H]) {
#pragma unroll
    for (int h = 0; h < H; ++h) {
#pragma unroll
        for (int d = 0; d < DPL; ++d) state.accumulator[h][d] += bias[h];
        bias[h] = 0.0f;
    }
}

}  // namespace

extern "C" __global__ void __launch_bounds__(WARPS * 32)
    attention_decode_k8v4_partial(SEISMIC_KERNEL_PARAMS) {
    const attention::Inputs in = ATTENTION_INPUTS();
    const attention::AffineHistory history{
        reinterpret_cast<u32 *>(SEISMIC_PTR(SEISMIC_BUFFER_HISTORY_KEY_CODES)),
        reinterpret_cast<u32 *>(SEISMIC_PTR(SEISMIC_BUFFER_HISTORY_KEY_COEFFICIENTS)),
        reinterpret_cast<u32 *>(SEISMIC_PTR(SEISMIC_BUFFER_HISTORY_VALUE_CODES)),
        reinterpret_cast<u32 *>(SEISMIC_PTR(SEISMIC_BUFFER_HISTORY_VALUE_COEFFICIENTS)),
        SEISMIC_HISTORY_KEY_CODES_STRIDE_0, SEISMIC_HISTORY_KEY_CODES_STRIDE_1,
        SEISMIC_HISTORY_KEY_COEFFICIENTS_STRIDE_0, SEISMIC_HISTORY_KEY_COEFFICIENTS_STRIDE_1,
        SEISMIC_HISTORY_VALUE_CODES_STRIDE_0, SEISMIC_HISTORY_VALUE_CODES_STRIDE_1,
        SEISMIC_HISTORY_VALUE_COEFFICIENTS_STRIDE_0, SEISMIC_HISTORY_VALUE_COEFFICIENTS_STRIDE_1,
        SEISMIC_PARAM_SLAB_ROWS};
    float *partials = reinterpret_cast<float *>(SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_PARTIALS));
    float *statistics = reinterpret_cast<float *>(SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_STATISTICS));
    const int kv = blockIdx.x;
    const int part = blockIdx.y;
    const u64 row = blockIdx.z;
    const int warp = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;

    extern __shared__ float shared[];
    float *queries = shared;                  // [G][W]
    float *exchange = queries + G * W;        // [WARPS][W]
    float *warp_stats = exchange + WARPS * W; // [WARPS][H][2]
    float *own = exchange + warp * W;

    const float query_scale = in.scale * attention::LOG2E;
    for (int h = warp; h < G; h += WARPS) {
        float x[DPL];
        attention::prepared_query(in, row, kv * G + h, x, own, lane);
#pragma unroll
        for (int d = 0; d < DPL; ++d) queries[h * W + lane * DPL + d] = x[d] * query_scale;
    }
    if (attention::FRESH && part == 0 && warp == WARPS - 1) {
        float k[DPL];
        attention::prepared_key(in, row, kv, k, own, lane);
        attention::append(in, history, row, kv, k, lane);
    }
    __syncthreads();

    const int head0 = (warp % SLICES) * H;
    float q[H][DPL];
    float qsum[H];
#pragma unroll
    for (int h = 0; h < H; ++h) {
        element::f32_span(queries + (head0 + h) * W + lane * DPL, q[h]);
        float sum = 0.0f;
#pragma unroll
        for (int d = 0; d < DPL; ++d) sum += q[h][d];
        qsum[h] = sum;
    }
    // This lane's (scale, zero) pair within a vector's pairs.
    const int pair = lane / attention::PAIR_LANES;

    // Equal partitions of the row's keys, then equal key-group ranges of this
    // one.
    const u64 spans = SEISMIC_DIM_R;
    const long long total = attention::visible_total(in, row, spans);
    const attention::Range partition =
        attention::partition(attention::Range{0, total}, (total + PARTS - 1) / PARTS, part);
    const attention::Range slice = attention::partition(
        partition, (partition.hi - partition.lo + GROUPS - 1) / GROUPS, warp / SLICES);

    State state;
    attention::clear(state);
    float bias[H];
#pragma unroll
    for (int h = 0; h < H; ++h) bias[h] = 0.0f;

    long long offset = 0;
    for (u64 span = 0; span <= spans && offset < slice.hi; ++span) {
        const attention::Span s = attention::span(in, row, span, spans);
        const long long length = s.hi > s.lo ? s.hi - s.lo : 0;
        const long long first = max(slice.lo, offset);
        const long long last = min(slice.hi, offset + length);
        if (first < last) {
            const int lo = s.lo + static_cast<int>(first - offset);
            const int hi = s.lo + static_cast<int>(last - offset);
            if (span < spans) {
                // Each slab's part of the span resolves its first rows once;
                // later rows are row strides past them (a per-token slab
                // lookup divides 64-bit rows).
                for (int part_lo = lo; part_lo < hi;) {
                    const int part_hi = history.slab_end(part_lo, hi);
                    const u32 *key_rows = history.key_row(part_lo, kv);
                    const u32 *value_rows = history.value_row(part_lo, kv);
                    const u32 *key_pairs = history.key_pair(part_lo, kv) + pair;
                    const u32 *value_pairs = history.value_pair(part_lo, kv) + pair;
                    const u64 key_step = history.key_codes_row;
                    const u64 value_step = history.value_codes_row;
                    const u64 key_pair_step = history.key_pairs_row / 2;
                    const u64 value_pair_step = history.value_pairs_row / 2;
                    for (int token = part_lo; token < part_hi; token += TOKENS) {
                        const int count = min(TOKENS, part_hi - token);
                        u32 k[TOKENS][KeyCodes::words];
                        u32 v[TOKENS][ValueCodes::words];
                        float2 kc[TOKENS];
                        float2 vc[TOKENS];
#pragma unroll
                        for (int t = 0; t < TOKENS; ++t) {
                            // Past the part, reload its last key: `absorb`
                            // skips it.
                            const u64 at = static_cast<u64>(min(token + t, part_hi - 1) - part_lo);
                            KeyCodes::load(key_rows + at * key_step, lane, k[t]);
                            ValueCodes::load(value_rows + at * value_step, lane, v[t]);
                            kc[t] = attention::coefficients(key_pairs + at * key_pair_step);
                            vc[t] = attention::coefficients(value_pairs + at * value_pair_step);
                        }
                        absorb(state, bias, q, qsum, k, kc, v, vc, count);
                    }
                    part_lo = part_hi;
                }
            } else {
                fold(state, bias);
                for (int token = lo; token < hi; ++token) {
                    float k[1][DPL];
                    float v[1][DPL];
                    attention::prepared_key(in, token, kv, k[0], own, lane);
                    attention::fresh_value(in, token, kv, v[0], lane);
                    float score[1][H];
                    attention::scores(q, k, score);
                    attention::absorb(state, score, v, 1);
                }
            }
        }
        offset += length;
    }
    fold(state, bias);
    attention::publish<WARPS, PARTS, SLICES>(state, exchange, warp_stats, partials, statistics, row,
                                             kv, part, warp, lane);
}

extern "C" __global__ void attention_decode_k8v4_merge(SEISMIC_KERNEL_PARAMS) {
    attention::decode_gate<PARTS>(
        ATTENTION_INPUTS(),
        reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_PARTIALS)),
        reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_STATISTICS)),
        SEISMIC_PTR(SEISMIC_RESULT_0_BUFFER), blockIdx.x, blockIdx.y, threadIdx.x);
}
