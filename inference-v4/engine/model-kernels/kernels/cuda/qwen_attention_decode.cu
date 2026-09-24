// qwen_attention_decode (M <= 8): split-KV decode attention.
//
// L1 `qwen_attention_decode_partial`, one block per (kv head, partition, row):
// the block prepares the row's G queries (rounded to the activation element
// as the contract publishes them, then scaled into the exp2 domain) and, in
// partition 0, appends the row's K/V. A row's tokens (its
// visible spans in order, then its fresh span) split into PARTS contiguous
// partitions; within a partition each warp scans a contiguous slice, lanes
// owning W / 32 dimensions, so every K/V row is read once for all G query
// heads of its kv head. Warp states merge in warp order into one partial
// (maximum, denominator, accumulator) per (row, query head, partition).
// L2 `qwen_attention_decode_merge`, one block per (query head, row): the
// partitions merge in partition order, then the sigmoid gate.

#include "common/attention.cuh"

namespace {

using attn::DPL;
using attn::G;
using attn::KV;
using attn::W;
using mx::u64;

constexpr int WARPS = SEISMIC_TUNE_WARPS;
constexpr int PARTS = SEISMIC_TUNE_PARTS;
// History tokens a warp keeps in flight (fewer when G query heads of state
// already fill the registers).
constexpr int TOKENS = G >= 8 ? 2 : 4;

struct State {
    float maximum[G];
    float denominator[G];
    float accumulator[G][DPL];
};

// Absorb `count` (1..N) keys with scores `score` (exp2 domain) and values `v`.
template <int N>
__device__ __forceinline__ void absorb(State &state, const float (&score)[N][G],
                                       const float (&v)[N][DPL], int count) {
#pragma unroll
    for (int h = 0; h < G; ++h) {
        float maximum = state.maximum[h];
#pragma unroll
        for (int t = 0; t < N; ++t)
            if (t < count) maximum = fmaxf(maximum, score[t][h]);
        // exp2(-inf) = 0 for the first keys of an empty state.
        const float carry = seismic_ex2_approx(state.maximum[h] - maximum);
        float denominator = state.denominator[h] * carry;
#pragma unroll
        for (int d = 0; d < DPL; ++d) state.accumulator[h][d] *= carry;
#pragma unroll
        for (int t = 0; t < N; ++t) {
            if (t < count) {
                const float p = seismic_ex2_approx(score[t][h] - maximum);
                denominator += p;
#pragma unroll
                for (int d = 0; d < DPL; ++d)
                    state.accumulator[h][d] = __fmaf_rn(p, v[t][d], state.accumulator[h][d]);
            }
        }
        state.denominator[h] = denominator;
        state.maximum[h] = maximum;
    }
}

template <int N>
__device__ __forceinline__ void scores(const float (&q)[G][DPL], const float (&k)[N][DPL],
                                       float (&score)[N][G]) {
#pragma unroll
    for (int t = 0; t < N; ++t) {
#pragma unroll
        for (int h = 0; h < G; ++h) {
            float dot = 0.0f;
#pragma unroll
            for (int d = 0; d < DPL; ++d) dot = __fmaf_rn(q[h][d], k[t][d], dot);
            score[t][h] = seismic_warp_sum_f32(dot);
        }
    }
}

}  // namespace

extern "C" __global__ void __launch_bounds__(WARPS * 32)
    qwen_attention_decode_partial(SEISMIC_KERNEL_PARAMS) {
    const attn::Inputs in = ATTN_INPUTS();
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

    const float query_scale = in.scale * attn::LOG2E;
    for (int h = warp; h < G; h += WARPS) {
        float x[DPL];
        mx::act_span<DPL, true>(in.query_gate, ATTN_QUERY_AT(row, kv * G + h) + lane * DPL, x);
        attn::norm_rotary(x, in.query_norm, SEISMIC_QUERY_NORM_STRIDE_0, in, row, own, lane);
#pragma unroll
        for (int d = 0; d < DPL; ++d) queries[h * W + lane * DPL + d] = mx::act_round(x[d]) * query_scale;
    }
    if (part == 0 && warp == WARPS - 1) {
        float k[DPL];
        attn::prepared_key(in, row, kv, k, own, lane);
        attn::append(in, row, kv, k, lane);
    }
    __syncthreads();

    float q[G][DPL];
#pragma unroll
    for (int h = 0; h < G; ++h) mx::f32_span(queries + h * W + lane * DPL, q[h]);

    const u64 spans = SEISMIC_DIM_R;
    long long total = 0;
    for (u64 span = 0; span <= spans; ++span) {
        const attn::Span s = attn::row_span(in, row, span, spans);
        total += s.hi > s.lo ? s.hi - s.lo : 0;
    }
    const long long per_part = (total + PARTS - 1) / PARTS;
    const long long part_lo = min(total, per_part * part);
    const long long part_hi = min(total, part_lo + per_part);
    const long long per_warp = (part_hi - part_lo + WARPS - 1) / WARPS;
    const long long warp_lo = min(part_hi, part_lo + per_warp * warp);
    const long long warp_hi = min(part_hi, warp_lo + per_warp);

    State state;
#pragma unroll
    for (int h = 0; h < G; ++h) {
        state.maximum[h] = -__int_as_float(0x7f800000);
        state.denominator[h] = 0.0f;
#pragma unroll
        for (int d = 0; d < DPL; ++d) state.accumulator[h][d] = 0.0f;
    }

    long long offset = 0;
    for (u64 span = 0; span <= spans && offset < warp_hi; ++span) {
        const attn::Span s = attn::row_span(in, row, span, spans);
        const long long length = s.hi > s.lo ? s.hi - s.lo : 0;
        const long long first = max(warp_lo, offset);
        const long long last = min(warp_hi, offset + length);
        if (first < last) {
            const int lo = s.lo + static_cast<int>(first - offset);
            const int hi = s.lo + static_cast<int>(last - offset);
            if (span < spans) {
                for (int token = lo; token < hi; token += TOKENS) {
                    const int count = min(TOKENS, hi - token);
                    float k[TOKENS][DPL];
                    float v[TOKENS][DPL];
#pragma unroll
                    for (int t = 0; t < TOKENS; ++t) {
                        if (t < count) {
                            mx::act_span<DPL, true>(
                                in.history_key, ATTN_HISTORY_KEY_AT(token + t, kv) + lane * DPL, k[t]);
                            mx::act_span<DPL, true>(
                                in.history_value, ATTN_HISTORY_VALUE_AT(token + t, kv) + lane * DPL,
                                v[t]);
                        } else {
#pragma unroll
                            for (int d = 0; d < DPL; ++d) k[t][d] = v[t][d] = 0.0f;
                        }
                    }
                    float score[TOKENS][G];
                    scores(q, k, score);
                    absorb(state, score, v, count);
                }
            } else {
                for (int token = lo; token < hi; ++token) {
                    float k[1][DPL];
                    float v[1][DPL];
                    attn::prepared_key(in, token, kv, k[0], own, lane);
                    mx::act_span<DPL, true>(in.value, ATTN_VALUE_AT(token, kv) + lane * DPL, v[0]);
                    float score[1][G];
                    scores(q, k, score);
                    absorb(state, score, v, 1);
                }
            }
        }
        offset += length;
    }

    // Merge the warps of the block in warp order, one query head at a time.
    if (lane == 0) {
#pragma unroll
        for (int h = 0; h < G; ++h) {
            warp_stats[(warp * G + h) * 2 + 0] = state.maximum[h];
            warp_stats[(warp * G + h) * 2 + 1] = state.denominator[h];
        }
    }
#pragma unroll
    for (int h = 0; h < G; ++h) {
        __syncthreads();
        mx::f32_span_store(exchange + warp * W + lane * DPL, state.accumulator[h]);
        __syncthreads();
        float maximum = -__int_as_float(0x7f800000);
        for (int w = 0; w < WARPS; ++w)
            if (warp_stats[(w * G + h) * 2 + 1] > 0.0f)
                maximum = fmaxf(maximum, warp_stats[(w * G + h) * 2]);
        const u64 at = (row * KV * G + kv * G + h) * PARTS + part;
        for (int column = threadIdx.x; column < W; column += WARPS * 32) {
            float accumulator = 0.0f;
            for (int w = 0; w < WARPS; ++w) {
                if (warp_stats[(w * G + h) * 2 + 1] > 0.0f) {
                    const float weight =
                        seismic_ex2_approx(warp_stats[(w * G + h) * 2] - maximum);
                    accumulator = __fmaf_rn(weight, exchange[w * W + column], accumulator);
                }
            }
            partials[at * W + column] = accumulator;
        }
        if (threadIdx.x == 0) {
            float denominator = 0.0f;
            for (int w = 0; w < WARPS; ++w) {
                const float l = warp_stats[(w * G + h) * 2 + 1];
                if (l > 0.0f)
                    denominator = __fmaf_rn(
                        l, seismic_ex2_approx(warp_stats[(w * G + h) * 2] - maximum), denominator);
            }
            statistics[at * 2 + 0] = maximum;
            statistics[at * 2 + 1] = denominator;
        }
    }
}

extern "C" __global__ void qwen_attention_decode_merge(SEISMIC_KERNEL_PARAMS) {
    const attn::Inputs in = ATTN_INPUTS();
    const float *partials =
        reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_PARTIALS));
    const float *statistics =
        reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_STATISTICS));
    mx::u8 *gated = SEISMIC_PTR(SEISMIC_RESULT_0_BUFFER);
    const int query_head = blockIdx.x;
    const u64 row = blockIdx.y;
    const int column = threadIdx.x;
    const u64 first = (row * KV * G + query_head) * PARTS;
    float maximum = -__int_as_float(0x7f800000);
    for (int part = 0; part < PARTS; ++part)
        if (statistics[(first + part) * 2 + 1] > 0.0f)
            maximum = fmaxf(maximum, statistics[(first + part) * 2]);
    float denominator = 0.0f;
    float accumulator = 0.0f;
    for (int part = 0; part < PARTS; ++part) {
        const float l = statistics[(first + part) * 2 + 1];
        if (l > 0.0f) {
            const float weight = seismic_ex2_approx(statistics[(first + part) * 2] - maximum);
            denominator = __fmaf_rn(l, weight, denominator);
            accumulator = __fmaf_rn(weight, partials[(first + part) * W + column], accumulator);
        }
    }
    const float gate = mx::act_load(in.query_gate, ATTN_GATE_AT(row, query_head) + column);
    const float attended = accumulator / fmaxf(denominator, 1e-30f) / (1.0f + expf(-gate));
    mx::act_store(gated,
                  row * SEISMIC_RESULT_0_STRIDE_0 + query_head * SEISMIC_RESULT_0_STRIDE_1 +
                      column * SEISMIC_RESULT_0_STRIDE_2,
                  attended);
}
