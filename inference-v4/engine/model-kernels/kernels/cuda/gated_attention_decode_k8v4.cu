// gated_attention_decode_k8v4 (M <= 8): `gated_attention_decode`'s split-KV
// decode over affine K8/V4 history.
//
// L1 `gated_attention_decode_k8v4_partial`, one block per (kv head, partition,
// row), partitioned exactly as the dense entry. History keys score as the sum
// over groups of scale * (q . code) + zero * sum(q) and values accumulate
// (p * scale) * code plus the bias sum(p * zero) of each lane's group, carried
// through the online softmax by the same factor as the accumulator
// (`absorb`), so no history element is decoded. The fresh span stays dense; the bias is folded into the
// accumulator before it. Partition 0 appends the row's key and value encoded.
// L2 `gated_attention_decode_k8v4_merge`: the dense entry's merge and gate.

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
// History keys a warp keeps in flight: affine rows are 2.6x smaller than
// dense ones, so twice the dense batch keeps as many bytes in flight.
constexpr int TOKENS = G >= 8 ? 4 : 8;

// Absorb `count` (1..N) affine-coded keys and values, with this lane's group
// pairs `kc` and `vc`, into `state` and this lane's per-head `bias` (exp2
// domain; `q` scaled into it, `qsum` the sums of this lane's dimensions).
template <int N>
__device__ __forceinline__ void absorb(attention::State &state, float (&bias)[G],
                                       const float (&q)[G][DPL], const float (&qsum)[G],
                                       const u32 (&k)[N][KeyCodes::words],
                                       const float2 (&kc)[N],
                                       const u32 (&v)[N][ValueCodes::words],
                                       const float2 (&vc)[N], int count) {
    float score[N][G];
#pragma unroll
    for (int t = 0; t < N; ++t) {
        float key[DPL];
#pragma unroll
        for (int d = 0; d < DPL; ++d) key[d] = KeyCodes::code(k[t], d);
#pragma unroll
        for (int h = 0; h < G; ++h) {
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
    for (int h = 0; h < G; ++h) {
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
__device__ __forceinline__ void fold(attention::State &state, float (&bias)[G]) {
#pragma unroll
    for (int h = 0; h < G; ++h) {
#pragma unroll
        for (int d = 0; d < DPL; ++d) state.accumulator[h][d] += bias[h];
        bias[h] = 0.0f;
    }
}

}  // namespace

extern "C" __global__ void __launch_bounds__(WARPS * 32)
    gated_attention_decode_k8v4_partial(SEISMIC_KERNEL_PARAMS) {
    const attention::Inputs in = ATTENTION_INPUTS();
    const attention::AffineHistory history{
        reinterpret_cast<u32 *>(SEISMIC_PTR(SEISMIC_BUFFER_HISTORY_KEY_CODES)),
        reinterpret_cast<u32 *>(SEISMIC_PTR(SEISMIC_BUFFER_HISTORY_KEY_COEFFICIENTS)),
        reinterpret_cast<u32 *>(SEISMIC_PTR(SEISMIC_BUFFER_HISTORY_VALUE_CODES)),
        reinterpret_cast<u32 *>(SEISMIC_PTR(SEISMIC_BUFFER_HISTORY_VALUE_COEFFICIENTS)),
        SEISMIC_HISTORY_KEY_CODES_STRIDE_0, SEISMIC_HISTORY_KEY_CODES_STRIDE_1,
        SEISMIC_HISTORY_KEY_COEFFICIENTS_STRIDE_0, SEISMIC_HISTORY_KEY_COEFFICIENTS_STRIDE_1,
        SEISMIC_HISTORY_VALUE_CODES_STRIDE_0, SEISMIC_HISTORY_VALUE_CODES_STRIDE_1,
        SEISMIC_HISTORY_VALUE_COEFFICIENTS_STRIDE_0, SEISMIC_HISTORY_VALUE_COEFFICIENTS_STRIDE_1};
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
    float *warp_stats = exchange + WARPS * W; // [WARPS][G][2]
    float *own = exchange + warp * W;

    const float query_scale = in.scale * attention::LOG2E;
    for (int h = warp; h < G; h += WARPS) {
        float x[DPL];
        element::span<Act, DPL, true>(in.query_gate, ATTENTION_QUERY_AT(row, kv * G + h) + lane * DPL, x);
        attention::norm_rotary(x, in.query_norm, SEISMIC_QUERY_NORM_STRIDE_0, in, row, own, lane);
#pragma unroll
        for (int d = 0; d < DPL; ++d) queries[h * W + lane * DPL + d] = Act::round(x[d]) * query_scale;
    }
    if (part == 0 && warp == WARPS - 1) {
        float k[DPL];
        attention::prepared_key(in, row, kv, k, own, lane);
        attention::append(in, history, row, kv, k, lane);
    }
    __syncthreads();

    float q[G][DPL];
    float qsum[G];
#pragma unroll
    for (int h = 0; h < G; ++h) {
        element::f32_span(queries + h * W + lane * DPL, q[h]);
        float sum = 0.0f;
#pragma unroll
        for (int d = 0; d < DPL; ++d) sum += q[h][d];
        qsum[h] = sum;
    }
    // This lane's (scale, zero) pair within a vector's pairs.
    const int pair = lane / attention::PAIR_LANES;

    // Equal partitions of the row's keys, then equal warp slices of this one.
    const u64 spans = SEISMIC_DIM_R;
    const long long total = attention::visible_total(in, row, spans);
    const attention::Range partition =
        attention::partition(attention::Range{0, total}, (total + PARTS - 1) / PARTS, part);
    const attention::Range slice = attention::partition(
        partition, (partition.hi - partition.lo + WARPS - 1) / WARPS, warp);

    attention::State state;
    attention::clear(state);
    float bias[G];
#pragma unroll
    for (int h = 0; h < G; ++h) bias[h] = 0.0f;

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
                for (int token = lo; token < hi; token += TOKENS) {
                    const int count = min(TOKENS, hi - token);
                    u32 k[TOKENS][KeyCodes::words];
                    u32 v[TOKENS][ValueCodes::words];
                    float2 kc[TOKENS];
                    float2 vc[TOKENS];
#pragma unroll
                    for (int t = 0; t < TOKENS; ++t) {
                        // Past the slice, reload its last key: `absorb` skips it.
                        const int at = min(token + t, hi - 1);
                        KeyCodes::load(history.key_row(at, kv), lane, k[t]);
                        ValueCodes::load(history.value_row(at, kv), lane, v[t]);
                        kc[t] = attention::coefficients(history.key_pair(at, kv) + pair);
                        vc[t] = attention::coefficients(history.value_pair(at, kv) + pair);
                    }
                    absorb(state, bias, q, qsum, k, kc, v, vc, count);
                }
            } else {
                fold(state, bias);
                for (int token = lo; token < hi; ++token) {
                    float k[1][DPL];
                    float v[1][DPL];
                    attention::prepared_key(in, token, kv, k[0], own, lane);
                    attention::fresh_value(in, token, kv, v[0], lane);
                    float score[1][G];
                    attention::scores(q, k, score);
                    attention::absorb(state, score, v, 1);
                }
            }
        }
        offset += length;
    }
    fold(state, bias);
    attention::publish<WARPS, PARTS>(state, exchange, warp_stats, partials, statistics, row, kv,
                                     part, warp, lane);
}

extern "C" __global__ void gated_attention_decode_k8v4_merge(SEISMIC_KERNEL_PARAMS) {
    attention::decode_gate<PARTS>(
        ATTENTION_INPUTS(),
        reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_PARTIALS)),
        reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_STATISTICS)),
        SEISMIC_PTR(SEISMIC_RESULT_0_BUFFER), blockIdx.x, blockIdx.y, threadIdx.x);
}
