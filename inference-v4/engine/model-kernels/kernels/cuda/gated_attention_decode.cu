// gated_attention_decode (M <= 8): split-KV decode attention.
//
// L1 `gated_attention_decode_partial`, one block per (kv head, partition, row):
// the block prepares the row's G queries (rounded to the activation element
// as the contract publishes them, then scaled into the exp2 domain) and, in
// partition 0, appends the row's K/V. A row's tokens (its
// visible spans in order, then its fresh span) split into PARTS contiguous
// partitions; within a partition each warp scans a contiguous slice, lanes
// owning W / 32 dimensions, so every K/V row is read once for all G query
// heads of its kv head. Warp states merge in warp order into one partial
// (maximum, denominator, accumulator) per (row, query head, partition).
// L2 `gated_attention_decode_merge`, one block per (query head, row): the
// partitions merge in partition order, then the sigmoid gate.

#include "lib/attention/attention.cuh"

namespace {

using attention::Act;
using attention::DPL;
using attention::G;
using attention::KV;
using attention::W;
using attention::u64;

constexpr int WARPS = SEISMIC_TUNE_WARPS;
constexpr int PARTS = SEISMIC_TUNE_PARTS;
// History tokens a warp keeps in flight (fewer when G query heads of state
// already fill the registers).
constexpr int TOKENS = G >= 8 ? 2 : 4;

}  // namespace

extern "C" __global__ void __launch_bounds__(WARPS * 32)
    gated_attention_decode_partial(SEISMIC_KERNEL_PARAMS) {
    const attention::Inputs in = ATTENTION_INPUTS();
    const attention::DenseHistory history{
        SEISMIC_PTR(SEISMIC_BUFFER_HISTORY_KEY), SEISMIC_PTR(SEISMIC_BUFFER_HISTORY_VALUE),
        SEISMIC_HISTORY_KEY_STRIDE_0, SEISMIC_HISTORY_KEY_STRIDE_1,
        SEISMIC_HISTORY_VALUE_STRIDE_0, SEISMIC_HISTORY_VALUE_STRIDE_1};
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
#pragma unroll
    for (int h = 0; h < G; ++h) element::f32_span(queries + h * W + lane * DPL, q[h]);

    // Equal partitions of the row's keys, then equal warp slices of this one.
    const u64 spans = SEISMIC_DIM_R;
    const long long total = attention::visible_total(in, row, spans);
    const attention::Range partition =
        attention::partition(attention::Range{0, total}, (total + PARTS - 1) / PARTS, part);
    const attention::Range slice = attention::partition(
        partition, (partition.hi - partition.lo + WARPS - 1) / WARPS, warp);

    attention::State state;
    attention::clear(state);

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
                    float k[TOKENS][DPL];
                    float v[TOKENS][DPL];
#pragma unroll
                    for (int t = 0; t < TOKENS; ++t) {
                        if (t < count) {
                            element::span<Act, DPL, true>(
                                history.key, history.key_at(token + t, kv) + lane * DPL, k[t]);
                            element::span<Act, DPL, true>(
                                history.value, history.value_at(token + t, kv) + lane * DPL, v[t]);
                        } else {
#pragma unroll
                            for (int d = 0; d < DPL; ++d) k[t][d] = v[t][d] = 0.0f;
                        }
                    }
                    float score[TOKENS][G];
                    attention::scores(q, k, score);
                    attention::absorb(state, score, v, count);
                }
            } else {
                for (int token = lo; token < hi; ++token) {
                    float k[1][DPL];
                    float v[1][DPL];
                    attention::prepared_key(in, token, kv, k[0], own, lane);
                    element::span<Act, DPL, true>(in.value, ATTENTION_VALUE_AT(token, kv) + lane * DPL, v[0]);
                    float score[1][G];
                    attention::scores(q, k, score);
                    attention::absorb(state, score, v, 1);
                }
            }
        }
        offset += length;
    }

    attention::publish<WARPS, PARTS>(state, exchange, warp_stats, partials, statistics, row, kv,
                                     part, warp, lane);
}

extern "C" __global__ void gated_attention_decode_merge(SEISMIC_KERNEL_PARAMS) {
    attention::decode_gate<PARTS>(
        ATTENTION_INPUTS(),
        reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_PARTIALS)),
        reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_STATISTICS)),
        SEISMIC_PTR(SEISMIC_RESULT_0_BUFFER), blockIdx.x, blockIdx.y, threadIdx.x);
}
